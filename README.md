# IMDb Sentiment Analysis and Rating Prediction

## Project Overview

This project performs sentiment analysis on IMDb movie reviews and predicts IMDb ratings using four different deep learning models:
1. LSTM (Long Short-Term Memory)
2. BERT (Bidirectional Encoder Representations from Transformers)
3. RoBERTa (Robustly Optimized BERT Approach)
4. XLNet (Generalized Autoregressive Pretraining for Language Understanding)

The project compares the accuracy of these models for both sentiment classification (positive/negative) and rating prediction (1-10 scale).

## Dataset

The dataset consists of:
- 50,000 IMDb movie reviews
- Binary sentiment labels (positive/negative)
- Actual IMDb ratings (1-10 scale)

Dataset source: [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Models Implemented

### 1. LSTM Model
- Traditional recurrent neural network with long short-term memory cells
- Handles sequential data effectively
- Baseline model for comparison

### 2. BERT (Bidirectional Encoder Representations from Transformers)
- State-of-the-art transformer-based model
- Pretrained on large corpus
- Fine-tuned for our specific tasks

### 3. RoBERTa
- Optimized version of BERT with improved training methodology
- Larger batch sizes and more training data
- Removes next-sentence prediction objective

### 4. XLNet
- Generalized autoregressive pretraining method
- Overcomes limitations of BERT's masked language modeling
- Considers all permutations of the factorization order

## Project Structure

```
imdb-sentiment-analysis/
├── data/
│   ├── raw/                # Raw dataset files
│   ├── processed/          # Processed and cleaned data
│   └── splits/             # Train/test/validation splits
├── models/
│   ├── lstm/               # LSTM model implementation
│   ├── bert/               # BERT model implementation
│   ├── roberta/            # RoBERTa model implementation
│   └── xlnet/              # XLNet model implementation
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_model_training.ipynb
│   └── 4_results_analysis.ipynb
├── scripts/
│   ├── preprocess.py       # Data preprocessing script
│   ├── train.py            # Model training script
│   └── evaluate.py         # Model evaluation script
├── results/
│   ├── metrics/            # Accuracy metrics for all models
│   ├── plots/              # Visualization plots
│   └── predictions/        # Sample predictions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data preprocessing:
```bash
python scripts/preprocess.py
```

2. Train models (example for BERT):
```bash
python scripts/train.py --model bert
```

Available model options: `lstm`, `bert`, `roberta`, `xlnet`

3. Evaluate models:
```bash
python scripts/evaluate.py --model all
```

## Results

The project will generate comparison metrics for:
- Sentiment classification accuracy
- Rating prediction mean absolute error (MAE)
- Rating prediction root mean squared error (RMSE)
- Training time comparison
- Inference speed comparison

## Dependencies

Python 3.8+

TensorFlow 2.0+

PyTorch

Transformers library

Pandas, NumPy, Matplotlib

Scikit-learn

- TensorFlow 2.0+:
An open-source machine learning framework developed by Google. It is used for building and training deep learning models, especially neural networks.

- PyTorch:
An open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and easy-to-use platform for deep learning research and production.

- Transformers Library (by Hugging Face):
A popular library providing pre-trained models and tools for Natural Language Processing (NLP) tasks like text classification, translation, and question answering.

- Pandas:
A powerful data analysis and manipulation library for Python. It provides data structures like DataFrames to work easily with structured data.

- NumPy:
A fundamental library for numerical computations in Python, supporting large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.

- Matplotlib:
A visualization library used for creating static, animated, and interactive plots and graphs in Python.

- Scikit-learn:
A machine learning library offering simple and efficient tools for data mining, data analysis, and building traditional ML models like classification, regression, and clustering.


## Future Work

- Implement ensemble methods combining predictions from multiple models
- Add more recent models like DeBERTa or ELECTRA
- Create a web interface for real-time predictions
- Expand to other review platforms (Rotten Tomatoes, Metacritic)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IMDb for making review data available
- Hugging Face for the Transformers library
- Original authors of BERT, RoBERTa, and XLNet papers