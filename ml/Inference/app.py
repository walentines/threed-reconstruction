import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../Train')

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from typing import List
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
import torch
import numpy as np
import cv2
import os
from Train.utils.util import *
from Train.pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
import torchvision.transforms as transforms
from PIL import Image
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

app = FastAPI()

pipeline = None
generator = None

# Initialize the pipeline
def init_pipeline(checkpoint):
    global pipeline, generator
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
    ).to('cuda:3')

    generator = torch.Generator(device='cuda:3').manual_seed(22211)

@app.on_event("startup")
def startup_event():
    checkpoint = '/home/sei2clj/threed-reconstruction/ml/Train/experiments/training_version_1_with_background/checkpoint-22800'
    init_pipeline(checkpoint)

pixel_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

@app.post("/generate_video/")
async def generate_video(prompt_image: UploadFile = File(...), canny_edges: List[UploadFile] = File(...)):
    global pipeline, generator
    
    if not pipeline or not generator:
        return JSONResponse(status_code=500, content={"error": "Pipeline is not initialized."})

    prompt_path = f"temp/{prompt_image.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(prompt_path, "wb") as buffer:
        shutil.copyfileobj(prompt_image.file, buffer)
    
    with torch.autocast(
        'cuda', enabled=True
    ):
    
        init_image = np.expand_dims(pil_image_to_numpy(Image.open(prompt_path)), axis=0)
        canny_edge_init = cv2.cvtColor(cv2.Canny(init_image[0], 50, 100), cv2.COLOR_GRAY2RGB)
        init_image = numpy_to_pt(init_image)
        init_pixel_values = pixel_transforms(init_image)
        
        control_images = []
        canny_edge_init = numpy_to_pt(np.expand_dims(canny_edge_init, 0))
        canny_edge_init_pixel_values = pixel_transforms(canny_edge_init)
        control_images.append(canny_edge_init_pixel_values.to('cuda'))
        
        for edge in canny_edges:
            edge_path = f"temp/{edge.filename}"
            with open(edge_path, "wb") as buffer:
                shutil.copyfileobj(edge.file, buffer)
            edge_image = np.expand_dims(pil_image_to_numpy(Image.open(edge_path)), axis=0)
            edge_image = numpy_to_pt(edge_image)
            edge_image = pixel_transforms(edge_image)
            control_images.append(edge_image.to('cuda'))
        
        all_canny_pixel_values = torch.stack(control_images)
        control_images = [validation_image.squeeze(0) for validation_image in all_canny_pixel_values.squeeze(0)]
        
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
        
        output_files = []
        for idx, frame in enumerate(video_frames[0][1:]):
            frame_path = f"temp/frame_{idx}.png"
            frame.save(frame_path)
            output_files.append(frame_path)
        
        return {"generated_frames": output_files}
