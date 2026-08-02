import argparse
import os
import cv2
import numpy as np
import time
from src.model import SeasonalMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "inference_result.png")
DEFAULT_IMG1 = os.path.join(BASE_DIR, "data", "processed", "val", "winter", "patch_0011.png")
DEFAULT_IMG2 = os.path.join(BASE_DIR, "data", "processed", "val", "summer", "patch_0011.png")

def draw_and_save_matches(img1_path, img2_path, results, output_path, max_draw=50):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1: w1 + w2] = img2

    mkpts0 = results['keypoints0']
    mkpts1 = results['keypoints1']
    inliers_mask = results['inliers_mask']

    total_matches = len(mkpts0)
    draw_count = min(total_matches, max_draw)
    indices_to_draw = np.linspace(0, total_matches - 1, draw_count, dtype=int)

    for i in indices_to_draw:
        x1, y1 = int(mkpts0[i][0]), int(mkpts0[i][1])
        x2, y2 = int(mkpts1[i][0]) + w1, int(mkpts1[i][1])

        if inliers_mask[i]:
            color = (0, 255, 0)
            thickness = 1
        else:
            color = (0, 0, 255)
            thickness = 1

        cv2.line(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.circle(vis, (x1, y1), 3, color, -1)
        cv2.circle(vis, (x2, y2), 3, color, -1)

    cv2.imwrite(output_path, vis)
    print(f"saved in {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, default=DEFAULT_IMG1)
    parser.add_argument("--img2", type=str, default=DEFAULT_IMG2)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="test_match path")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    
    args = parser.parse_args()
    
    print(f"1. {args.img1}\n2. {args.img2}")
    img1_abs = os.path.abspath(args.img1)
    img2_abs = os.path.abspath(args.img2)
    output_abs = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    if not os.path.exists(img1_abs):
        print(img1_abs)
        return
        
    if not os.path.exists(img2_abs):
        print(img2_abs)
        return
    
    matcher = SeasonalMatcher(conf_threshold=args.conf)
    start_time = time.time()
    results = matcher.match(img1_abs, img2_abs)
    elapsed = time.time() - start_time
    
    n_matches = results["num_matches"]
    n_inliers = results["num_inliers"]
    ratio = (n_inliers / n_matches * 100) if n_matches > 0 else 0.0
    
    report = f"""
    ================================
    Matches: {n_matches}
    Inliers:     {n_inliers}
    Inlier Ratio:             {ratio:.2f}%
    Runtime:            {elapsed:.2f} сек.
    ================================
    """
    print(report)
    
    if n_matches > 0:
        draw_and_save_matches(img1_abs, img2_abs, results, output_abs)
    else:
        print("no matches. Image is not saved")

if __name__ == "__main__":
    main()