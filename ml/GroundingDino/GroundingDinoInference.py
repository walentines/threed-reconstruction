import torch
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

class GroundingDinoInference:
    def __init__(self, model_id, device="cuda:2"):
        self.model_id = model_id
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
    
    def run_inference(self, image_path, text):
        image = Image.open(image_path)        

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        width, height = image.size
        postprocessed_outputs = self.processor.image_processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.3)
        results = postprocessed_outputs[0]
        
        return self.find_largest_area_bounding_box(results).cpu().numpy()
    
    def find_largest_area_bounding_box(self, results):
        idx_max_area = 0
        max_area = 0.0
        for i, box in enumerate(results['boxes']):
            area_box = abs(box[2] - box[0]) * abs(box[3] - box[1])
            if area_box > max_area:
                idx_max_area = i
                max_area = area_box

        box = results['boxes'][idx_max_area]

        return box

