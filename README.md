# Fake Job Posting Detection

This project implements a deep learning model to detect fake job postings. The model uses natural language processing techniques and neural networks to classify job postings as either real or fake.

## Project Structure

- `data/`: Directory for storing the dataset
- `models/`: Directory for saving trained models
- `src/`: Source code directory
  - `data_processing.py`: Data loading and preprocessing
  - `model.py`: Neural network architecture
  - `train.py`: Training script
  - `evaluate.py`: Evaluation script
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in the `data/` directory
2. Train the model:
```bash
python src/train.py
```
3. Evaluate the model:
```bash
python src/evaluate.py
```

## Model Architecture

The model uses a combination of:
- TF-IDF vectorization for text processing
- Neural network with multiple layers for classification
- Binary cross-entropy loss function

## Dataset

The project uses a dataset of job postings labeled as either real or fake. The dataset should be in CSV format with the following columns:
- `title`: Job title
- `description`: Job description
- `label`: Binary label (0 for real, 1 for fake) 