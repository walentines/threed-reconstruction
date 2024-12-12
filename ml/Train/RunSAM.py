import sys
import os
import torch
import cv2
from tqdm import tqdm

sys.path.append('..')

from SegmentAnything.SAMInference import SamInference
from GroundingDino.GroundingDinoInference import GroundingDinoInference

grounding_dino_inference = GroundingDinoInference('IDEA-Research/grounding-dino-base', device='cuda:0')
sam_inference = SamInference(grounding_dino_inference, 'path_to_weights', 'vit_h', device='cuda:0')

DIR = 'your_dir'


if __name__ == '__main__':
    dirs = os.listdir(DIR)
    for dir in dirs:
        full_dir = os.path.join(DIR, dir)
        images = [file for file in os.listdir(full_dir) if file.startswith('frame_') and file.endswith('.jpg') and not file.endswith('_canny_edge.jpg') and not file.endswith('_segmentation.jpg')]
        for image_path in tqdm(images):
            image_path = os.path.join(full_dir, image_path)
            segmentation_path = image_path.split('.')[0] + '_segmentation.jpg'
            if not os.path.exists(segmentation_path):
                with torch.no_grad():
                    segmentation = sam_inference.run_inference(image_path, 'car.')
                cv2.imwrite(segmentation_path, segmentation)




