# deep_learning_MBFProject
## Protein Function Prediction Using CNNs and ESM-2
---

In this project, we tried to predict protein function directly from the amino acid sequence using deep learning. Specifically, we tried to predict **Gene Ontology (GO) terms** — a standardised way of describing what a protein does.

We did this in two stages:
1. First we tested our models on **simulated data** where we knew exactly what the answer was (a known motif was planted in the sequence)
2. Then we applied the best models to **real human protein data** to predict five different GO terms

---

## Repository Structure

```
📁 project/
│
├── DL_MBFProject_simulated.ipynb       
├── DL_GO_Data_MBFProject.ipynb     
├── Weight and biases report on simulated data
├──Weight and biases report on real data
└── README.md                      
```

---

## Notebook 1 — Simulated Data (`DL_MBFProject_simulated.ipynb`)

This notebook covers everything related to the six simulated datasets.

**What it does, step by step:**

1. Clones the project data from the WUR GitLab repository
2. Loads and explores the simulated sequence files (`.seq`) and label files (`.pos`)
3. Splits the data into train (70%), validation (15%), and test (15%) sets
4. Implements two CNN architectures:
   - **Embedding CNN** — learns a dense vector for each amino acid
   - **One-hot CNN** — uses a fixed binary representation for each amino acid
5. Runs a **hyperparameter sweep** over dropout (0.1, 0.3, 0.5) and learning rate (1e-4, 1e-3, 3e-3)
6. Runs a **subsampling experiment** — what happens if you remove half the positives or negatives from training?
7. Scales the best model to **all six simulated datasets** and reports results

**Key results:**
- Learning rate matters far more than dropout
- Both encodings achieve similar final accuracy (~0.99 on easy datasets)
- Removing positive examples collapses model performance — class balance is critical
- 4 out of 6 datasets reach above 0.79 test accuracy

---

## Notebook 2 — GO Term Prediction (`DL_GO_Data_MBFProject.ipynb`)

This notebook covers the real human protein data and GO term prediction.

**What it does, step by step:**

1. Loads 6,784 annotated human proteins and five GO term label files
2. Handles **severe class imbalance** using weighted BCE loss (`pos_weight`)
3. Trains a **1D CNN** baseline on all five GO terms simultaneously
4. Extracts embeddings from **frozen ESM-2 8M and 35M** models (via HuggingFace)
5. Trains a lightweight **MLP classification head** on top of the ESM-2 embeddings
6. Evaluates using **mAUPRC, mAUROC, and micro-F1** (appropriate for imbalanced data)
7. Builds a simple **ensemble** (average of CNN + ESM-2 35M predictions)
8. **Optimises decision thresholds** per GO term on the validation set
9. Runs the best model on **14,765 unlabeled proteins** and saves predictions to CSV

**Key results:**

| Model | mAUPRC | mAUROC | micro-F1 |
|---|---|---|---|
| CNN | 0.2306 | 0.7398 | 0.1996 |
| ESM-2 8M | 0.2750 | 0.7965 | 0.2033 |
| ESM-2 35M | 0.3074 | 0.8040 | 0.2391 |
| Ensemble | 0.3110 | 0.8016 | 0.2815 |

- Transmembrane transport was the easiest GO term to predict (strong local sequence signal)
- Signal transduction and negative regulation of apoptosis were very hard (context-dependent)
- The CNN actually beat ESM-2 on mitochondrion — local motifs can favour simpler models

---

## How to Run

Both notebooks are designed to run on **Google Colab**. Just open them and run all cells from top to bottom.

> **Note:** The ESM-2 embedding extraction in Notebook 2 can take 10–20 minutes on CPU. Using a GPU runtime in Colab will speed this up significantly.

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Model building and training |
| `transformers` | Loading ESM-2 from HuggingFace |
| `scikit-learn` | Metrics (AUPRC, AUROC, F1) and data splitting |
| `pandas` | Results tables and CSV export |
| `numpy` | Array operations |
| `matplotlib` | Training curves and bar charts |
| `wandb` | Experiment tracking (optional, can be disabled) |

---

## Output Files

After running Notebook 2, you will find:

- **`go_predictions_14765.csv`** — predictions for all 14,765 unlabeled human proteins. Contains probability scores and binary predictions (0/1) for each of the five GO terms:
  - `mitochondrion`
  - `signal_transduction`
  - `extracellular`
  - `transmembrane`
  - `neg_apoptosis`

---
