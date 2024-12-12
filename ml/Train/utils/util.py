import os
import imageio
import numpy as np
from typing import Union
import cv2
import sys

sys.path.append('..')

import torch
import torchvision
import torch.distributed as dist
from PIL import Image
import datetime

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from SegmentAnything.SAMInference import SamInference
from GroundingDino.GroundingDinoInference import GroundingDinoInference
from diffusers.training_utils import EMAModel
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from models.controlnet_sdv import ControlNetSDVModel
import math
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
import transformers
import diffusers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from Dataset.RealCarDataset import RealCarDataset

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255

def sample_images(car_images, start_frame, num_frames):
    indices = np.linspace(start_frame, len(car_images), num=num_frames, dtype=np.int32, endpoint=False)

    return [car_images[i] for i in indices]

def validation_images_control(validation_path, sam_inference):
    current_video = [os.path.join(validation_path, file) for file in os.listdir(validation_path) if file.startswith('frame_') and file.endswith('.jpg') and not file.endswith('_canny_edge.jpg') and not file.endswith('_segmentation.jpg')]
    current_video = [image_path for image_path in sample_images(current_video, 0, 14)]

    # Using canny-edge as prompt (saving canny edge for next epochs and upcoming trainings)
    current_video_canny_edge = []
    for image in current_video:
        canny_edge_path = image.split('.')[0] + '_canny_edge.jpg'
        if not os.path.exists(canny_edge_path):
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            canny_edge = cv2.Canny(img, 50, 150)
            cv2.imwrite(canny_edge_path, canny_edge)
        current_video_canny_edge.append(canny_edge_path)

    # Compute segmentation mask (saving it for next epochs and upcoming trainings)
    current_video_segmentation_masks = []
    for image in current_video:
        segmentation_path = image.split('.')[0] + '_segmentation.jpg'
        if not os.path.exists(segmentation_path):
            segmentation = sam_inference.run_inference(image, 'car.')
            cv2.imwrite(segmentation_path, segmentation)
        else:
            segmentation = cv2.imread(segmentation_path, -1)
        current_video_segmentation_masks.append(segmentation)
    segmentation_masks = torch.tensor(np.array(current_video_segmentation_masks))

    # Load image frames
    numpy_images = np.array([pil_image_to_numpy(Image.open(img)) for img in current_video])
    pixel_values = numpy_to_pt(numpy_images)
    segmented_pixel_values = torch.where(segmentation_masks.unsqueeze(1).repeat(1, 3, 1, 1) == 255, pixel_values, 0)

    # Load canny frames
    numpy_canny_images = np.array([pil_image_to_numpy(Image.open(canny)) for canny in current_video_canny_edge])
    canny_pixel_values = numpy_to_pt(numpy_canny_images)
    segmented_canny_pixel_values = torch.where(segmentation_masks.unsqueeze(1).repeat(1, 3, 1, 1) == 255, canny_pixel_values, 0)

    pixel_values_list = []
    for pixel_value in segmented_pixel_values:
        pixel_values_list.append(pixel_value)

    canny_pixel_values_list = []
    for pixel_value in segmented_canny_pixel_values:
        canny_pixel_values_list.append(pixel_value)

    del segmentation_masks
    torch.cuda.empty_cache()

    return pixel_values_list, canny_pixel_values_list

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

def save_model_hook(models, weights, output_dir):
    for i, model in enumerate(models):
        model.save_pretrained(os.path.join(output_dir, "controlnet"))

        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()
    
def load_model_hook(models, input_dir):
    for i in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        load_model = ControlNetSDVModel.from_pretrained(
            input_dir, subfolder="controlnet")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n

def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def make_dataset(config, sam, mode):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if mode == 'training':
        dataset = RealCarDataset(config.dataloader.training.csv_path,config.dataloader.training.video_folder,config.dataloader.training.condition_folder, sam=sam, sample_n_frames=config.dataloader.training.sample_n_frames)
    elif mode == 'validation':
        dataset = RealCarDataset(config.dataloader.validation.csv_path,config.dataloader.validation.video_folder,config.dataloader.validation.condition_folder, sam=sam, sample_n_frames=config.dataloader.validation.sample_n_frames, sample_n_times=config.dataloader.validation.sample_n_times, validation=True)
    else:
        raise RuntimeError('Please choose between training and validation')
    return dataset

def setup_xformers(config, logger, unet):
    if config.trainer.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")
    
    return unet

def instantiate_and_configure_models(config, accelerator, logger):
    feature_extractor = CLIPImageProcessor.from_pretrained(
            config.model.pretrained_model_name_or_path, subfolder="feature_extractor"
            )
        
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config.model.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
        )
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config.model.pretrained_model_name_or_path, subfolder="vae", variant="fp16"
        )
    
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
            config.model.pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
            variant="fp16",
        )
    
    if config.model.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetSDVModel.from_pretrained(config.model.controlnet_model_name_or_path, subfolder="controlnet")
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetSDVModel.from_unet(unet)
    
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    return feature_extractor, image_encoder, vae, unet, controlnet, weight_dtype

def setup_accelerator(config, logger):
    logging_dir = os.path.join(config.trainer.output_dir, config.trainer.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.trainer.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=config.trainer.gradient_accumulation_steps, 
                                mixed_precision=config.trainer.mixed_precision,
                                project_config=accelerator_project_config)
    
    generator = torch.Generator(device=accelerator.device).manual_seed(config.trainer.seed)

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config.trainer.seed is not None:
        set_seed(config.trainer.seed)
    
    if accelerator.is_main_process:
        if config.trainer.output_dir is not None:
            os.makedirs(config.trainer.output_dir, exist_ok=True)
    
    return accelerator, generator

def encode_image(pixel_values, feature_extractor, image_encoder):
    pixel_values = pixel_values * 2.0 - 1.0
    pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
    pixel_values = (pixel_values + 1.0) / 2.0

    # Normalize the image with for CLIP input
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values.to(image_encoder.device)

    image_embeddings = image_encoder(pixel_values).image_embeds
    image_embeddings= image_embeddings.unsqueeze(1)
    return image_embeddings

def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def save_combined_frames(batch_output, validation_images, validation_control_images,output_folder):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    validation_images = [torchvision.transforms.functional.to_pil_image((validation_image * 255).type(torch.uint8)) for validation_image in validation_images]
    validation_control_images = [torchvision.transforms.functional.to_pil_image((validation_control_image * 255).type(torch.uint8)) for validation_control_image in validation_control_images]

    # Combine frames into a list without converting (since they are already PIL Images)
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3  # adjust number of columns as needed
    rows = (num_images + cols - 1) // cols
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"combined_frames_{timestamp}.png"
    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols)
    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)
    
    # Now define the full path for the file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    output_loc = os.path.join(output_folder, filename)
    
    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")

def get_add_time_ids(
    fps,
    motion_bucket_ids,  # Expecting a list of tensor floats
    noise_aug_strength,
    dtype,
    batch_size,
    unet=None,
    device=None,  # Add a device parameter
):
    # Determine the target device
    target_device = device if device is not None else 'cpu'

    # Ensure motion_bucket_ids is a tensor and on the target device
    if not isinstance(motion_bucket_ids, torch.Tensor):
        motion_bucket_ids = torch.tensor(motion_bucket_ids, dtype=dtype, device=target_device)
    else:
        motion_bucket_ids = motion_bucket_ids.to(device=target_device)

    # Reshape motion_bucket_ids if necessary
    if motion_bucket_ids.dim() == 1:
        motion_bucket_ids = motion_bucket_ids.view(-1, 1)

    # Check for batch size consistency
    if motion_bucket_ids.size(0) != batch_size:
        raise ValueError("The length of motion_bucket_ids must match the batch_size.")

    # Create fps and noise_aug_strength tensors on the target device
    add_time_ids = torch.tensor([fps, noise_aug_strength], dtype=dtype, device=target_device).repeat(batch_size, 1)

    # Concatenate with motion_bucket_ids
    add_time_ids = torch.cat([add_time_ids, motion_bucket_ids], dim=1)

    # Checking the dimensions of the added time embedding
    passed_add_embed_dim = unet.config.addition_time_embed_dim * add_time_ids.size(1)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
            f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
            "Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    return add_time_ids
    

# def load_weights(
#     animation_pipeline,
#     # motion module
#     motion_module_path         = "",
#     motion_module_lora_configs = [],
#     # image layers
#     dreambooth_model_path = "",
#     lora_model_path       = "",
#     lora_alpha            = 0.8,
# ):
#     # 1.1 motion module
#     unet_state_dict = {}
#     if motion_module_path != "":
#         print(f"load motion module from {motion_module_path}")
#         motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
#         motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
#         unet_state_dict.update({name.replace("module.", ""): param for name, param in motion_module_state_dict.items()})
    
#     missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
#     assert len(unexpected) == 0
#     del unet_state_dict

#    # if dreambooth_model_path != "":
#    #     print(f"load dreambooth model from {dreambooth_model_path}")
#   #      if dreambooth_model_path.endswith(".safetensors"):
#   #          dreambooth_state_dict = {}
#   #          with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
#    #             for key in f.keys():
#    #                 dreambooth_state_dict[key.replace("module.", "")] = f.get_tensor(key)
#    #     elif dreambooth_model_path.endswith(".ckpt"):
#    #         dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
#    #         dreambooth_state_dict = {k.replace("module.", ""): v for k, v in dreambooth_state_dict.items()}
            
#         # 1. vae
#     #    converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
#     #    animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
#         # 2. unet
#     #    converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
#     #    animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
#         # 3. text_model
#      #   animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
#      #   del dreambooth_state_dict
        
#     if lora_model_path != "":
#         print(f"load lora model from {lora_model_path}")
#         assert lora_model_path.endswith(".safetensors")
#         lora_state_dict = {}
#         with safe_open(lora_model_path, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 lora_state_dict[key.replace("module.", "")] = f.get_tensor(key)
                
#         animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
#         del lora_state_dict

#     for motion_module_lora_config in motion_module_lora_configs:
#         path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
#         print(f"load motion LoRA from {path}")

#         motion_lora_state_dict = torch.load(path, map_location="cpu")
#         motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
#         motion_lora_state_dict = {k.replace("module.", ""): v for k, v in motion_lora_state_dict.items()}

#         animation_pipeline = convert_motion_lora_ckpt_to_diffusers(animation_pipeline, motion_lora_state_dict, alpha)

#     return animation_pipeline
