import torch
import numpy as np
import kornia as K
import cv2
from kornia.feature import LoFTR

class SeasonalMatcher:
    def __init__(self, pretrained="outdoor", conf_threshold=0.2, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.conf_threshold = conf_threshold

        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()
    
    def preprocess_image(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError
        
        img_tensor = K.image_to_tensor(img, keepdim=False).float() / 255.0
        return img_tensor.to(self.device)