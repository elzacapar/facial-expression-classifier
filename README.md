# Facial Expression Classifier

A deep learning project that classifies human facial expressions into 7 emotion 
categories using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

Built as part of a Data Science education assignment. The assignment was given in Swedish 
but the project is documented in English to make it accessible as a portfolio piece.

## Overview

The model classifies grayscale 48x48 face images into one of 7 emotions:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

**Final model performance:** 52.8% test accuracy (random baseline: 14.3%)

## Project Structure

facial-expression-classifier/
├── facial_expression_classifier.ipynb  # main notebook — code + analysis
├── model/
│   ├── facial_expression_model.keras   # saved model
│   ├── history3.json                   # training history (iteration 3)
│   └── history3b.json                  # training history (iteration 3b)
├── requirements.txt
└── README.md

## What the Notebook Covers

1. **Data exploration** — class distribution, sample images, data quality observations
2. **Data preparation** — normalization, train/validation split, pipeline setup
3. **Model building** — CNN architecture with Conv2D, BatchNorm, Dropout layers
4. **Training** — 3 iterations with documented reasoning, early stopping, training curves
5. **Evaluation** — accuracy, F1-score, confusion matrix, per-class analysis
6. **Predictions** — inference on unseen images with confidence scores
7. **Analysis** — overfitting diagnosis, limitation discussion, class imbalance
8. **Reflection** — lessons learned, what would be done differently

## Key Findings

- Class imbalance significantly affects performance — `disgust` (436 images) is never 
  predicted correctly while `happy` (7,215 images) achieves F1: 0.77
- Iteration 3 (BatchNorm + Dropout + Augmentation) reduced the train/val accuracy gap 
  from 18% to 6%, indicating better generalization
- Visually similar emotions (`fear`, `sad`, `neutral`) are frequently confused

## Dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — not included in this 
repository due to size (60MB). Download and place in:

C:/your/path/FER-2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
└── ...

Update the `TRAIN_DIR` and `TEST_DIR` paths in the notebook accordingly.

## Setup

```bash
git clone https://github.com/elzacapar/facial-expression-classifier
cd facial-expression-classifier
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Then open `facial_expression_classifier.ipynb` in VS Code or Jupyter.

## Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Pillow