import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../Train')
import gradio as gr
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
import torch
from Train.utils.util import *
from Train.pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
import torchvision.transforms as transforms
from hydra import compose, initialize
from PIL import Image
import numpy as np

# Initialize your models
def init_pipeline(checkpoint):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder", variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    controlnet = ControlNetSDVModel.from_pretrained(checkpoint, subfolder="controlnet")

    # Move models to half precision
    image_encoder.to(dtype=torch.float16)
    vae.to(dtype=torch.float16)
    unet.to(dtype=torch.float16)
    controlnet.eval()

    # Initialize the pipeline
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        unet=unet,
        controlnet=controlnet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=torch.float16,
    ).to('cuda')

    generator = torch.Generator(device='cuda:3').manual_seed(22211)

    return pipeline, generator

# Inference function
def generate_video(prompt_image, canny_edges):
    cfg = None
    with initialize(version_base=None, config_path="../Train/hydra_config"):
        cfg = compose(config_name="train_config")
    checkpoint = cfg.inference.checkpoint_path  # Path to your checkpoint
    pipeline, generator = init_pipeline(checkpoint)
    pixel_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


    control_images = []
    # Load the initial prompt image
    init_image = np.expand_dims(pil_image_to_numpy(Image.open(prompt_image)), axis=0)
    init_image = numpy_to_pt(init_image)
    init_pixel_values = pixel_transforms(init_image)

    # Load mock canny edge for prompt image
    canny_edge_init = np.expand_dims(pil_image_to_numpy(Image.open(prompt_image)), axis=0)
    canny_edge_init = numpy_to_pt(canny_edge_init)
    canny_edge_init_pixel_values = pixel_transforms(canny_edge_init)
    control_images.append(canny_edge_init_pixel_values)
    # Load canny edge images
    for edge_path in canny_edges:
        edge_image = np.expand_dims(pil_image_to_numpy(Image.open(edge_path)), axis=0)
        edge_image = numpy_to_pt(edge_image)
        edge_image = pixel_transforms(edge_image)
        control_images.append(edge_image.to('cuda').squeeze())
    all_canny_pixel_values = torch.stack(control_images)
    control_images = [validation_image.squeeze(0) for validation_image in all_canny_pixel_values.squeeze(0)]
    # Run the pipeline
    with torch.no_grad():
        video_frames = pipeline(
                    init_pixel_values, 
                    control_images,
                    height=512,
                    width=512,
                    num_frames=len(control_images),
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=2,
                    noise_aug_strength=0.0,
                    min_guidance_scale=1.0,
                    max_guidance_scale=1.0000000001,
                    num_inference_steps=50,
                    generator=generator
                ).frames

    # Convert frames to video (as a GIF)
    video_frames[0][0].save(
        "output.gif", save_all=True, append_images=video_frames[0][1:], duration=100, loop=0
    )
    return "output.gif"

# Build Gradio Interface
interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Sketchpad(label="Sketchpad (or upload a prompt image)", brush_radius=5),
        gr.Files(label="Canny Edge Prompts (Up to 14)", file_types=["image"]),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Stable Video Diffusion Interface with Sketchpad",
    description="Draw an initial image on the sketchpad or upload a prompt image, and upload up to 14 canny edge images to generate a 360° video using Stable Video Diffusion.",
)

# Launch Gradio
interface.launch()
