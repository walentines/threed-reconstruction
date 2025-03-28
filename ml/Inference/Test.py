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
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Test:
    def __init__(self, config, checkpoint):
        self.config = config
        self.dataset = make_dataset(config, None, 'test', depth=True)
        print(len(self.dataset))
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
        image_encoder.to(dtype=torch.float32)
        vae.to(dtype=torch.float32)
        unet.to(dtype=torch.float32)
        controlnet = ControlNetSDVModel.from_pretrained(checkpoint, subfolder="controlnet")
        self.generator = torch.Generator(device='cuda:2').manual_seed(config.trainer.seed)
        self.pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
                            self.config.model.pretrained_model_name_or_path,
                            unet=unet,
                            controlnet=controlnet,
                            image_encoder=image_encoder,
                            vae=vae,
                            torch_dtype=torch.float32,
                        ).to('cuda:2')
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0,
        )

        self.test_fid = 0.0
        self.test_inception_score_mean = 0.0
        self.test_inception_score_std = 0.0

        self.fid = FrechetInceptionDistance(feature=2048)
        self.inception = InceptionScore(splits=3)
    
    def __compute_fid(self, true_images, fake_images):
        self.fid.update(true_images, real=True)
        self.fid.update(fake_images, real=False)
        return self.fid.compute()

    def __compute_inception_score(self, images):
        self.inception.update(images)
        return self.inception.compute()
    
    def __test_step(self, batch, step):
         test_save_dir = os.path.join('test_results_background', f"test_images_{step}")

         if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)
         with torch.autocast(
            'cuda', enabled=True
            ):
            images = [validation_image.squeeze(0) for validation_image in batch['pixel_values'].squeeze(0)]
            control_images = [validation_image.squeeze(0) for validation_image in batch['depth_pixel_values'].squeeze(0)]
            segmentation_masks = torch.stack(batch['segmentation_masks'], axis=0)
            segmentation_masks = torch.nn.functional.interpolate(
                segmentation_masks,
                (512, 512),
                mode='bilinear'
            )
            del batch['pixel_values']
            del batch['depth_pixel_values']
            with torch.no_grad():
                video_frames = self.pipeline(
                    images[0], 
                    control_images,
                    height=self.config.dataloader.validation.height,
                    width=self.config.dataloader.validation.width,
                    num_frames=2,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=2,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.0,
                    max_guidance_scale=3.0,
                    num_inference_steps=50,
                    generator=self.generator
                ).frames
            images = torch.stack(images)
            images = torch.where(segmentation_masks.repeat(1, 3, 1, 1) == 255, images, -1)
            pred_images = torch.stack([pil_to_tensor(img) for sublist in video_frames for img in sublist])
            pred_images = torch.where(segmentation_masks.repeat(1, 3, 1, 1) == 255, pred_images, 0)
            fid = self.__compute_fid(((images * 0.5 + 0.5) * 255).type(torch.uint8), pred_images.type(torch.uint8))
            print(fid)
            self.test_fid += fid
            inception_score = self.__compute_inception_score(pred_images)
            inception_score_mean = inception_score[0]
            self.test_inception_score_mean += inception_score_mean
            inception_score_std = inception_score[1]
            self.test_inception_score_std += inception_score_std

            save_combined_frames(pred_images, images, control_images, test_save_dir)
            del images
            del control_images

    def __test(self):
        for step, batch in enumerate(self.test_dataloader):
            self.__test_step(batch, step)

    
    def __call__(self):
        self.__test()
        print(f'FID: {self.test_fid / len(self.dataset)}')
        print(f'Inception Score Mean: {self.test_inception_score_mean / len(self.dataset)}')
        print(f'Inception Score STD: {self.test_inception_score_std / len(self.dataset)}')


        
@hydra.main(version_base=None, config_path="../Train/hydra_config", config_name="config_hydra")
def main(config):
    Test(config, config.inference.checkpoint_path)()

if __name__ == '__main__':
    main()