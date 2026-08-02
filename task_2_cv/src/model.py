import torch
import numpy as np
import kornia as K
import cv2
from kornia.feature import LoFTR
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

    def match(self, img1_path, img2_path):

        img0_tensor = self.preprocess_image(img1_path)
        img1_tensor = self.preprocess_image(img2_path)
        input_dict = {"image0": img0_tensor, "image1": img1_tensor}

        with torch.no_grad():
            correspondes = self.matcher(input_dict)

        mkpts0 = correspondes['keypoints0'].cpu().numpy()
        mkpts1 = correspondes['keypoints1'].cpu().numpy()
        confidence = correspondes['confidence'].cpu().numpy()

        confidence_mask = confidence >= self.conf_threshold
        mkpts0 = mkpts0[confidence_mask]
        mkpts1 = mkpts1[confidence_mask]
        confidence = confidence[confidence_mask]

        inliers_mask = np.zeros(len(mkpts0), dtype=bool)
        if len(mkpts0) >= 4:
            H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            if inliers is not None:
                inliers_mask = inliers.ravel().astype(bool)

        return {"keypoints0": mkpts0,
                "keypoints1": mkpts1,
                "confidence": confidence,
                "inliers_mask": len(inliers_mask),
                "num_matches": len(mkpts0),
                "num_inliers": np.sum(len(inliers_mask))
                }