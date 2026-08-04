#  Task 2: Sentinel-2 image matching

This module implements a Computer Vision pipeline designed to find keypoint correspondences and match aerial/satellite image patches across drastic seasonal changes (e.g., Winter | Summer) using deep feature matching techniques.

---

## 📁 Repository Structure

```
task_2_cv/
├── data/
│   ├── processed/
│   │   ├── train
│   │   └── val    
│   └── raw/
├── dataset_creation.ipynb
├── demo.ipynb                  
├── train.py                    # Fine-tuning training script
├── src/
│   └──model.py           
├── inference.py                # Inference wrapper class & CLI
├── mountains_text.txt
└── README.md                   # Project documentation
```
##  Architecture & Approach

* **Core Model:**  — a deep feature extraction and matching pipeline optimized to handle extreme domain shifts caused by snow cover, shadows, and seasonal vegetation changes.
* **Dataset Preparation Pipeline:**
  * **Sliding Window:** Slices high-resolution imagery into 512x512 patches using a 256px overlap stride.
  * **NoData Removal:** Automatically drops empty/black border areas where mean brightness is under 25.
  * **Cloud & Snow Masking:** Filters out uninformative patches where over 40% of the pixels consist of featureless snow or dense cloud cover.
* **Inference Pipeline:**
  * Calculates key quantitative metrics: **Total Matches**, **Inlier Count**, **Inlier Ratio (%)**, and **Runtime**.
  * Generates visualization outputs highlighting **Inliers (Green)** vs. **Outliers (Red)**.
  * Features robust absolute path resolution and auto-creation of output directories.

---

##  Model Weights & Dataset

* **Dataset:** https://drive.google.com/drive/u/0/folders/1s3CzAmhDFefi_JAnactIUTuXwf3A1ZOJ
* **Model Weights:** https://drive.google.com/drive/u/0/folders/1mOf2fQcnd1IHeQqFD67i_BD4IBvx6XaR

> **Note:** If custom model weights are needed, download them from the link above and place them into the `weights/` directory before executing local training or evaluation.

---

##  Quick Start / Local Setup:

### 1. Installation
Install the project dependencies:
```bash
pip install -r requirements.txt

1. Download saved_model.zip from the link above.

2. Extract the archive contents directly into task_1_ner/saved_model/.

3. Verify that config.json, model.safetensors (or pytorch_model.bin), and tokenizer.json are present in the directory.
```
### 2. Run Inference CLI

To run matching using default relative paths:
```bash
python inference.py
```
### 3. Interactive Demo
To run the end-to-end demo and inspect step-by-step visual results, launch Jupyter Notebook and run:
```bash
jupyter notebook demo.ipynb
```