#  Machine Learning Assessment Solutions

Welcome! This repository contains structured, modular, and fully documented solutions for two machine learning challenges: **NLP** and **Computer Vision**.

---

## 📁 Repository Structure
```
├── task_1_ner/           # Task 1: Named Entity Recognition for Mountains
│   ├── data/             # Training / validation datasets
│   ├── saved_model/      # Fine-tuned model weights
│   ├── train.py          # Fine-tuning training pipeline
│   ├── inference.py      # Inference CLI & wrapper
│   ├── demo.ipynb        # Interactive demonstration notebook
│   └── README.md         # Detailed Task 1 documentation
│
├── task_2_cv/            # Task 2: Seasonal Satellite Image Matching
│   ├── data/             # Raw & processed 512x512 patch datasets
│   ├── src/              # SeasonalMatcher model definition
│   ├── results/          # Generated match visualization outputs
│   ├── inference.py      # Matching CLI & visualization engine
│   ├── demo.ipynb        # Interactive demonstration notebook
│   └── README.md         # Detailed Task 2 documentation
│
└── README.md             # Repository overview (You are here)
```
##  Projects Overview

###  Task 1: Mountain Named Entity Recognition (NER)
* **Goal:** Extract and reconstruct mountain names from unstructured English text.
* **Approach:** Fine-tuned **DistilBERT / BERT** architecture utilizing BIO (Begin, Inside, Outside) tagging for token classification, complemented by custom subword prefix reconstruction logic.
* **Key Features:** Full data preparation, dataset generation, training script, CLI inference, and interactive notebook demo.
* **Documentation & Setup:** See [`task_1_ner/README.md`](./task_1_ner/README.md).

---

###  Task 2: Seasonal Satellite Image Matching
* **Goal:** Detect keypoint correspondences and match aerial/satellite image patches across severe domain shifts (e.g., **Winter vs. Summer**).
* **Approach:** Built custom `SeasonalMatcher` leveraging **LoFTR** (Local Feature TRansformer) and RANSAC homography estimation.
* **Data Pipeline:** Includes automatic patch slicing (512x512), empty border removal (mean brightness filtering), and heavy cloud/snow mask filtering (>40%).
* **Documentation & Setup:** See [`task_2_cv/README.md`](./task_2_cv/README.md).

---

## 🚀 Quick Start

To run or evaluate either project, navigate into the respective folder and follow its dedicated `README.md`:
