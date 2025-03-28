import sys
import os
import torch
import cv2
from tqdm import tqdm
import numpy as np

sys.path.append('..')

from DepthAnything.DepthAnythingInference import DepthAnythingInference

DIR = 'path/to/dir'
depth_anything = DepthAnythingInference('../DepthAnything/weights/depth_anything_v2_vitl.pth')

if __name__ == '__main__':
    dirs = os.listdir(DIR)
    for dir in dirs:
        full_dir = os.path.join(DIR, dir)
        print(full_dir)
        images = [file for file in os.listdir(full_dir) if file.startswith('frame_') and file.endswith('.jpg') and not file.endswith('_canny_edge.jpg') and not file.endswith('_segmentation.jpg') and not file.endswith('_depth.jpg')]
        for image_path in tqdm(images):
            image_path = os.path.join(full_dir, image_path)
            depth_path = image_path.split('.')[0] + '_depth.jpg'
            print(depth_path)
            if not os.path.exists(depth_path):
                with torch.no_grad():
                    depth = depth_anything.run_inference(image_path)
                if depth is not None:
                    cv2.imwrite(depth_path, (depth / depth.max() * 255).astype(np.uint8))