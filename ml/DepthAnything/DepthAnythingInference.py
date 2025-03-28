import cv2
from depth_anything_v2.dpt import DepthAnythingV2
import torch

class DepthAnythingInference:
    def __init__(self, model_path):
        self.__model_path = model_path
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

        encoder = 'vitl' 
        self.__model = DepthAnythingV2(**model_configs[encoder])
        self.__model.load_state_dict(torch.load(self.__model_path, map_location='cpu'))
        self.__model = self.__model.to('cuda:0').eval()
    
    def run_inference(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        depth = self.__model.infer_image(image)

        return depth
