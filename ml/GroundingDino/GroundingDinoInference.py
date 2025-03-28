import torch
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GroundingDinoInference:
    def __init__(self, model_id, device="cuda:2"):
        self.model_id = model_id
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
    
    def run_inference(self, image_path, text):
        try:
            image = Image.open(image_path)  
        except:
            return None      

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        width, height = image.size
        postprocessed_outputs = self.processor.image_processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.3)
        results = postprocessed_outputs[0]
        bboxes = self.find_largest_area_bounding_box(results)
        if bboxes is None:
            return None
        return bboxes.cpu().numpy()
    
    def find_largest_area_bounding_box(self, results):
        idx_max_area = 0
        max_area = 0.0
        for i, box in enumerate(results['boxes']):
            area_box = abs(box[2] - box[0]) * abs(box[3] - box[1])
            if area_box > max_area:
                idx_max_area = i
                max_area = area_box
        if len(results['boxes']) == 0:
            return None
        box = results['boxes'][idx_max_area]

        return box

    def plot_bounding_box(self, image_path, bboxes):
        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Convert bounding box to integer values
        x_min, y_min, x_max, y_max = map(int, bboxes)

        # Draw the bounding box on the image (color is red, thickness is 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Display the image with the bounding box
        cv2.imwrite("output.jpg", image)
