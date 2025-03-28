import torch

class PredictNextFrame:
    def __init__(self, pipeline, checkpoint_dir, height, width):
        self.pipeline = pipeline
        self.checkpoint_dir = checkpoint_dir
        self.height = height
        self.width = width
    
    def predict_next_frame(self, prompt_image, canny_edge_1, canny_edge_2):
        canny_edge_controls = [torch.tensor(canny_edge_1), torch.tensor(canny_edge_2)]
        video_frames = self.pipeline(
                    prompt_image, 
                    canny_edge_controls,
                    height=self.height,
                    width=self.width,
                    num_frames=2,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=2,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.0,
                    max_guidance_scale=3.0,
                    num_inference_steps=50,
                ).frames

        return video_frames

