import os, io, csv, math, random
import numpy as np
import pandas as pd
from einops import rearrange
import sys
import logging
from tqdm import tqdm

import torch
from decord import VideoReader
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

sys.path.append('..')

from SegmentAnything.SAMInference import SamInference
from GroundingDino.GroundingDinoInference import GroundingDinoInference


logger = logging.getLogger('RealCarDataset')
logging.basicConfig()
logger.setLevel('INFO')

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

class RealCarDataset(Dataset):
    def __init__(self, csv_path, images_folder, control_folder, sam_checkpoint, sam=None, validation=False, model_type='vit_h', grounding_dino_model_id='IDEA-Research/grounding-dino-base', device='cuda:1', sample_size=256, sample_stride=10, sample_n_frames=5, sample_n_times=4):
        with open(csv_path, 'r') as csvfile:
            self.dataset = pd.read_csv(csvfile)

        self.images_folder = images_folder
        self.control_folder = control_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.canny_pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size),
        ])
        if sam is None:
            grounding_dino_inference = GroundingDinoInference(grounding_dino_model_id)

            self.sam_inference = SamInference(grounding_dino_inference, sam_checkpoint, model_type, device)
        else:
            self.sam_inference = sam
        self.validation=validation

        self.__preprocess_dataset(sample_n_times)

    def __sample_images(self, car_images, start_frame):
        indices = np.linspace(start_frame, len(car_images), num=self.sample_n_frames, dtype=np.int32, endpoint=False)

        return [car_images[i] for i in indices]

    def __len__(self):
        return len(self.postprocessed_dataset)

    def __preprocess_dataset(self, num_samples):
        self.postprocessed_dataset = []
        logger.info('Preprocessing dataset...')
        for dir in tqdm(self.dataset["videoid"].tolist()):
            car_images = sorted([os.path.join(dir, file) for file in os.listdir(os.path.join(self.images_folder, dir)) if file.startswith('frame_') and file.endswith('.jpg') and not file.endswith('_canny_edge.jpg') and not file.endswith('_segmentation.jpg')])
            if len(car_images) == 0:
                logger.warning(f'Skipping {dir} since it contains no images...')
                continue
            for i in range(num_samples):
                images = [os.path.join(self.images_folder, image_path) for image_path in self.__sample_images(car_images, (i * 10) % len(car_images))]
                self.postprocessed_dataset.append(images)
        logger.info('Dataset preprocessed!')

    def __getitem__(self, idx):
        current_video = self.postprocessed_dataset[idx]

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
                with torch.no_grad():
                    segmentation = self.sam_inference.run_inference(image, 'car.')
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

        del segmentation_masks
        torch.cuda.empty_cache()
        segmented_canny_pixel_values = self.canny_pixel_transforms(segmented_canny_pixel_values)
        if self.validation:
            segmented_pixel_values = self.canny_pixel_transforms(segmented_pixel_values)
            segmented_pixel_values = [pixel_value.squeeze() for pixel_value in segmented_pixel_values]
            segmented_canny_pixel_values = [pixel_value.squeeze() for pixel_value in segmented_canny_pixel_values]
        else:
            segmented_pixel_values = self.pixel_transforms(segmented_pixel_values)
        sample = dict(pixel_values=segmented_pixel_values, depth_pixel_values=segmented_canny_pixel_values)

        return sample


# if __name__ == '__main__':
#     dataset = RealCarDataset('test.csv', '/mnt/hddmount1/sei2clj/HQ339', '/mnt/hddmount1/sei2clj/HQ339', '/home/sei2clj/threed-reconstruction/ml/SegmentAnything/segment-anything/weights/sam_vit_h_4b8939.pth')
#     dataset.__getitem__(0)