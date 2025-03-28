import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../Train')
from Train.utils.util import *
from Train.pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
import hydra
import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"]="3"

class Generate360:
    def __init__(self, config, checkpoint):
        self.config = config
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
        image_encoder.to(dtype=torch.float16)
        vae.to(dtype=torch.float16)
        unet.to(dtype=torch.float16)
        controlnet = ControlNetSDVModel.from_pretrained(checkpoint, subfolder="controlnet")
        self.generator = torch.Generator(device='cuda:3').manual_seed(config.trainer.seed)
        controlnet.eval()
        self.pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
                            self.config.model.pretrained_model_name_or_path,
                            unet=unet,
                            controlnet=controlnet,
                            image_encoder=image_encoder,
                            vae=vae,
                            torch_dtype=torch.float16,
                        ).to('cuda:3')
        self.images_folder = config.dataloader.training.video_folder

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def generate_360(self, dir, init_image_path):
        car_images = sorted([os.path.join(dir, file) for file in os.listdir(os.path.join(self.images_folder, dir)) if file.startswith('frame_') and file.endswith('.jpg') and not file.endswith('_canny_edge.jpg') and not file.endswith('_segmentation.jpg') and os.path.exists(os.path.join(self.images_folder, dir, file.split('.')[0] + '_segmentation.jpg'))])
        car_images = car_images[1:]
        if not os.path.exists(dir):
            os.mkdir(dir)
        init_image_path = os.path.join(self.images_folder, init_image_path)
        num_frames = 13
        with torch.autocast(
            'cuda', enabled=True
        ):
            for i in range(0, len(car_images), num_frames):
                # INIT IMAGES
                init_image = np.expand_dims(pil_image_to_numpy(Image.open(init_image_path)), axis=0)
                init_image = numpy_to_pt(init_image)

                segmentation_path = os.path.join(self.images_folder, init_image_path.split('.')[0] + '_segmentation.jpg')
                segmentation_mask_init_image = cv2.imread(segmentation_path, -1)
                if segmentation_mask_init_image.shape != (1440, 1920):
                    segmentation_mask_init_image = np.transpose(segmentation_mask_init_image, (1, 0))
                segmentation_mask_init_image = torch.tensor(segmentation_mask_init_image)
                init_image = torch.where(segmentation_mask_init_image.unsqueeze(0).repeat(1, 3, 1, 1) == 255, init_image, 0)

                init_image_canny = np.expand_dims(pil_image_to_numpy(Image.open(os.path.join(self.images_folder, dir, init_image_path.split('/')[-1].split('.')[0] + '_canny_edge.jpg'))), axis=0)
                init_image_canny = numpy_to_pt(init_image_canny)
                init_image_canny = torch.where(segmentation_mask_init_image.unsqueeze(0).repeat(1, 3, 1, 1) == 255, init_image_canny, 0)

                init_pixel_values = self.pixel_transforms(init_image)
                init_canny_pixel_values = self.pixel_transforms(init_image_canny)

                car_image_subset = car_images[i:i + num_frames]
                all_canny_pixel_values = [init_canny_pixel_values.squeeze()]
                for car_image in car_image_subset:
                    # CANNY PROMPT
                    car_image_canny_full = os.path.join(self.images_folder, car_image.split('.')[0] + '_canny_edge.jpg')
                    car_image_seg_full = os.path.join(self.images_folder, car_image.split('.')[0] + '_segmentation.jpg')
                    segmentation_mask_curr_image = cv2.imread(car_image_seg_full, -1)
                    if segmentation_mask_curr_image.shape != (1440, 1920):
                        segmentation_mask_curr_image = np.transpose(segmentation_mask_curr_image, (1, 0))
                    segmentation_mask_curr_image = torch.tensor(segmentation_mask_curr_image)
                    curr_canny = np.expand_dims(pil_image_to_numpy(Image.open(car_image_canny_full)), axis=0)
                    curr_canny = numpy_to_pt(curr_canny)
                    curr_canny = torch.where(segmentation_mask_curr_image.unsqueeze(0).repeat(1, 3, 1, 1) == 255, curr_canny, 0)
                    
                    canny_pixel_values = self.pixel_transforms(curr_canny)
                    all_canny_pixel_values.append(canny_pixel_values.squeeze())
                all_canny_pixel_values = torch.stack(all_canny_pixel_values)
                control_images = [validation_image.squeeze(0) for validation_image in all_canny_pixel_values.squeeze(0)]
                
                with torch.no_grad():
                    video_frames = self.pipeline(
                        init_pixel_values, 
                        control_images,
                        height=self.config.dataloader.validation.height,
                        width=self.config.dataloader.validation.width,
                        num_frames=len(control_images),
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=2,
                        noise_aug_strength=0.0,
                        min_guidance_scale=1.0,
                        max_guidance_scale=3.0,
                        num_inference_steps=50,
                        generator=self.generator
                    ).frames
                    for j, frame in enumerate(video_frames[0][1:]):
                        frame = frame.resize((1920, 1440))
                        frame.save(os.path.join(dir, car_images[i + j].split('/')[-1]))
                    # init_image_path = os.path.join(dir, car_images[i + num_frames - 1].split('/')[-1])
                    


@hydra.main(version_base=None, config_path="../Train/hydra_config", config_name="config_hydra")
def main(config):
    Generate360(config, config.inference.checkpoint_path).generate_360('2024_04_09_16_34_06', '2024_04_09_16_34_06/frame_00000.jpg')

if __name__ == '__main__':
    main()