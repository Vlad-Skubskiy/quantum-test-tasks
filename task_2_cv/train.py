import os
import glob
import numpy as np
import time
from src.model import SeasonalMatcher

def evaluate_pipeline(data_dir="data/processed/val"):
    winter_files = sorted(glob.glob(os.path.join(data_dir, "winter", "*.png")))
    summer_files = sorted(glob.glob(os.path.join(data_dir, "summer", "*.png")))
    if not winter_files or not summer_files:
        return

    matcher = SeasonalMatcher(conf_threshold=0.2)

    total_matches = []
    total_inliers = []
    inlier_ratios = []

    start_time = time.time()

    for i, (w_path, s_path) in enumerate(zip(winter_files, summer_files)):

        if os.path.basename(w_path) != os.path.basename(s_path):
            print(f"{w_path} and {s_path} have different names")
            continue

        results = matcher.match(s_path, w_path)
        n_matches = results["num_matches"]
        n_inliers=  results["num_inliers"]
        ratio = (n_inliers / n_matches) if n_matches > 0 else 0.0

        total_matches.append(n_matches)
        total_inliers.append(n_inliers)
        inlier_ratios.append(ratio)

        if (i + 1) % 10 == 0 or (i + 1) == len(winter_files):
            print(f"matched {i + 1}/{len(winter_files)} files")

    elapsed_time = time.time() - start_time
    report = f"""
    =======================================
    Time runned:           {elapsed_time:.1f} сек.
    Pairs matched:             {len(total_matches)}
    Mean Matches:     {np.mean(total_matches):.1f}
    Mean Inliers:     {np.mean(total_inliers):.1f}
    Mean Inlier Ratio:       {np.mean(inlier_ratios) * 100:.2f}%
    =======================================
    """
    print(report)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(BASE_DIR, "data", "processed", "val")

    evaluate_pipeline(data_dir=target_data_dir)