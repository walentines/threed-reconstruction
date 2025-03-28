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
from PIL import Image
import numpy as np
import cv2
import os

pipeline = None
generator = None

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
    ).to('cuda:3')

    generator = torch.Generator(device='cuda:3').manual_seed(22211)

    return pipeline, generator


# Inference function
def generate_video(prompt_image, *canny_edges):
    global pipeline, generator
    checkpoint = '/home/sei2clj/threed-reconstruction/ml/Train/training_version_1_with_background/checkpoint-9120'
    if pipeline is None and generator is None:
        pipeline, generator = init_pipeline(checkpoint)
    
    pixel_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    with torch.autocast('cuda', enabled=True):
        control_images = []

        # Load the initial prompt image
        init_image = np.expand_dims(pil_image_to_numpy(Image.open(prompt_image)), axis=0)
        canny_edge_init = cv2.cvtColor(cv2.Canny(init_image[0], 50, 100), cv2.COLOR_GRAY2RGB)
        init_image = numpy_to_pt(init_image)
        init_pixel_values = pixel_transforms(init_image)

        # Load mock Canny edge for prompt image
        canny_edge_init = numpy_to_pt(np.expand_dims(canny_edge_init, 0))
        canny_edge_init_pixel_values = pixel_transforms(canny_edge_init)
        control_images.append(canny_edge_init_pixel_values.to('cuda'))

        # Load Canny edge images
        for edge_path in canny_edges:
            if not edge_path or os.stat(edge_path).st_size == 0:
                continue
            edge_image = np.expand_dims(pil_image_to_numpy(Image.open(edge_path)), axis=0)
            edge_image = numpy_to_pt(edge_image)
            edge_image = pixel_transforms(edge_image)
            control_images.append(edge_image.to('cuda'))

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

        # Return video frames as separate images
        return [frame for frame in video_frames[0][1:]]

def preview_canny_edges(canny_edge_files):
    images = []
    for file in canny_edge_files:
        if file is not None:
            image = Image.open(file).resize((128, 128))  # Resize for grid display
            images.append(image)
    return images

# Build Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("## Stable Video Diffusion with Canny Edge Images")

    prompt_image = gr.Image(label="Prompt Image", type="filepath")
    num_edges_slider = gr.Slider(label="Number of Canny Edge Images", minimum=1, maximum=14, step=1, value=7)
     # Dynamic Grid for Canny Edge inputs
    grid = []
    with gr.Row():
        for i in range(3):
            with gr.Column():
                for j in range(4):
                    grid.append(gr.Image(label=f"Canny Edge {i * 3 + j + 1}", type="filepath", visible=False, height=128, width=128))
    with gr.Row():
        with gr.Column():
            grid.append(gr.Image(label=f"Canny Edge {13}", type="filepath", visible=False, height=128, width=128))
        with gr.Column():
            grid.append(gr.Image(label=f"Canny Edge {14}", type="filepath", visible=False, height=128, width=128))
    def update_canny_inputs(num_inputs):
        updates = [gr.update(visible=(i < num_inputs)) for i in range(14)]
        return updates
    

    generate_button = gr.Button("Generate Video")
    output_frames = gr.Gallery(label="Generated Frames", type="pil")

    num_edges_slider.change(
        fn=update_canny_inputs,
        inputs=[num_edges_slider],
        outputs=grid
    )

    # Connect the components
    generate_button.click(
        fn=generate_video,
        inputs=[prompt_image] + grid,
        outputs=output_frames
    )

# Launch Gradio
interface.launch()
