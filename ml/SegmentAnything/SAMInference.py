import sys
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
sys.path.append('..')
from GroundingDino.GroundingDinoInference import GroundingDinoInference

class SamInference:
    def __init__(self, grounding_dino, sam_checkpoint, model_type, device):
        self.grounding_dino = grounding_dino
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
    
    def run_inference(self, image_path, text):
        bounding_box = self.grounding_dino.run_inference(image_path, text)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)

        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bounding_box[None, :],
            multimask_output=False,
            )

        return masks[0].astype(np.uint8) * 255

# if __name__ == '__main__':
#     grounding_dino = GroundingDinoInference('IDEA-Research/grounding-dino-base', 'cuda')
#     sam_inference = SamInference(grounding_dino, 'segment-anything/weights/sam_vit_h_4b8939.pth', 'vit_h', 'cuda')
#     sam_inference.run_inference('/mnt/hddmount1/sei2clj/HQ339/2024_01_18_15_41_12/frame_00986.jpg', 'car.')


        

