#  Task 1: Mountain Named Entity Recognition (NER)

This module implements a Named Entity Recognition (NER) pipeline designed to identify mountain names in English texts using a Transformer architecture (**DistilBERT / BERT**).

---

## 📁 Repository Structure

```
task_1_ner/
├── data/
│   ├── raw_mountains.csv       # Source list of target entities
│   ├── train_ner_dataset.json  # Training dataset with BIO tagging
│   └── val_ner_dataset.json    # Validation dataset
├── saved_model/                # Fine-tuned model weights & config
├── dataset_creation.ipynb      # Notebook detailing dataset generation & EDA
├── demo.ipynb                  # Demo notebook
├── train.py                    # Fine-tuning training script
├── inference.py                # Inference wrapper class & CLI
├── mountains_text.txt          # Testing text
└── README.md                   # Project documentation
```
## Architecture & Approach


Base Model: distilbert-base-uncased — a lightweight, efficient Transformer model for token classification (TokenClassification).

* **Annotation Scheme:** BIO (Begin, Inside, Outside) format:
  * **B-MOUNTAIN:** Start of a mountain entity (e.g., Mount in Mount Everest).
  * **I-MOUNTAIN:** Continuation of a mountain entity (e.g., Everest in Mount Everest).
  * **O:** Non-entity tokens and punctuation.

* **Post-Processing:** The `inference.py` script features custom subword reconstruction logic (`##` prefixes) to seamlessly join tokens back into natural phrases with appropriate spacing.
## Model Weights


https://drive.google.com/drive/u/0/folders/1523x1jXWroNxn5gIx70Xj6w7BIcT60gH

### Local Installation Steps:

1. Download saved_model.zip from the link above.

2. Extract the archive contents directly into task_1_ner/saved_model/.

3. Verify that config.json, model.safetensors (or pytorch_model.bin), and tokenizer.json are present in the directory.
