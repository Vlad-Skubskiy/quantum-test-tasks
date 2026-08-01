import torch
import numpy as np
import kornia as k
from kornia.feature import LoFTR

class SeaasonalMatcher:
    def __init__(self, pretrained="outdoor", conf_threshold=0.2, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.conf_threshold = conf_threshold

        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()
    