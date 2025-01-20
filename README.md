# Sentiment Analysis Project

## Overview
The Sentiment Analysis Project leverages natural language processing (NLP) and machine learning techniques to identify, extract, and analyze sentiments from textual data. This project focuses on analyzing product reviews from e-commerce platforms to determine whether sentiments expressed are positive, negative, or neutral. The goal is to provide insights into consumer opinions and market trends, helping businesses improve their products and strategies.

## Metadata
- **Project Name**: Sentiment Analysis of Product Reviews
- **Author**: David Satria Alamsyah
- **Dataset Source**: [Kaggle Flipkart Product Reviews](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)
- **Technologies**: Python, TensorFlow, Hugging Face Transformers (BERT), Pandas, NumPy, Scikit-learn, PyTorch, NLTK
- **Models**: CNN, Attention-based BiGRU, BERT

## Tools and Technologies
1. **Python**: General-purpose programming language used for rapid development and integration of machine learning models.
2. **TensorFlow**: Framework for building and training deep learning models such as CNN and BiGRU.
3. **Hugging Face Transformers (BERT)**: Pretrained language model used for advanced NLP tasks.
4. **Pandas**: Library for data manipulation and analysis.
5. **NumPy**: Library for numerical computations.
6. **Scikit-learn**: Toolkit for machine learning algorithms and model evaluation.
7. **PyTorch**: Framework for flexible and efficient model training.
8. **NLTK**: Library for text preprocessing tasks like tokenization, stemming, and lemmatization.

## Project Objectives
1. Automate sentiment analysis of product reviews.
2. Provide insights into consumer opinions and preferences.
3. Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

## Scope
- Develop a sentiment analysis model for automatic evaluation of product reviews.
- Utilize product review datasets from Kaggle.
- Evaluate model performance for reliability and validity.

## Dataset
The dataset contains information about product reviews, including:
- **Columns**: Product Name, Price, Rating, Review, Summary, Sentiment
- **Size**: 205,053 rows and 6 columns
- **Source**: Flipkart e-commerce platform

[Dataset Link](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)

## Methodology
1. **Approaches**:
   - Sentiment Lexicon: Classify texts using predefined word sentiment labels (positive, negative, neutral).
   - Deep Learning: Use advanced models such as CNN, attention-based BiGRU, and BERT.

2. **Implementation Steps**:
   - Data Preprocessing: Clean and normalize text using NLTK.
   - Model Training: Train deep learning models using TensorFlow and PyTorch.
   - Evaluation: Assess model performance with Scikit-learn metrics.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/MacOS
   env\Scripts\activate   # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset using Pandas.
2. Preprocess the text data using NLTK.
3. Train and evaluate the models (CNN, BiGRU, BERT) using the provided scripts.
4. Generate insights and visualize results.

---
Feel free to customize the script to suit your project needs.
