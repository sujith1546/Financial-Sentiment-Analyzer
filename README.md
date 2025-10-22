# Financial Sentiment Analysis using FinBERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete implementation for fine-tuning FinBERT on Indian Financial News dataset for sentiment analysis. This project fine-tunes the ProsusAI/finbert model on a dataset of Indian financial news articles to classify sentiment as positive, neutral, or negative.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Model Inference](#model-inference)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project demonstrates how to fine-tune the FinBERT model on Indian financial news data to perform sentiment analysis. The model is trained to classify financial text into three sentiment categories:
- Positive (2)
- Neutral (1)
- Negative (0)

The implementation includes:
- Data loading and preprocessing
- Model fine-tuning with Hugging Face Transformers
- Evaluation with comprehensive metrics
- Error analysis
- Model inference functionality

## Features

- 📊 Comprehensive data exploration and visualization
- 🧹 Robust data preprocessing pipeline
- 🤖 Fine-tuning of state-of-the-art FinBERT model
- 📈 Detailed performance metrics and visualizations
- 🔍 Error analysis for model improvement insights
- 💾 Model saving and loading functionality
- 🚀 Ready-to-use inference function

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install datasets transformers torch pandas numpy scikit-learn matplotlib seaborn accelerate
```

## Project Structure

```
financial-sentiment-analysis/
├── finbert_sentiment_analysis.py    # Main implementation script
├── requirements.txt                 # Package dependencies
├── README.md                        # This file
├── finbert_indian_finance/          # Fine-tuned model (after training)
└── results/                         # Training outputs and logs
```

## Usage

### 1. Run the complete training and evaluation pipeline:

```bash
python finbert_sentiment_analysis.py
```

### 2. Using the trained model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = './finbert_indian_finance'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    return {
        'sentiment': ['negative', 'neutral', 'positive'][predicted_class],
        'confidence': confidence
    }

text = "The company reported record profits in the last quarter."
result = predict_sentiment(text)
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.2f}")
```

## Results

After fine-tuning FinBERT on the Indian Financial News dataset, the model achieves the following performance metrics:

📊 **FINAL MODEL PERFORMANCE METRICS**

- **Dataset**: Indian Financial News (26,000+ articles)
- **Model**: FinBERT (Fine-tuned)
- **Training Samples**: 18,872
- **Validation Samples**: 4,044
- **Test Samples**: 4,045
- **Training Time**: 0:32:16.427674

🎯 **Test Set Results:**
- **Accuracy**: 87.52%
- **Precision**: 87.74%
- **Recall**: 87.52%
- **F1-Score**: 87.59%

📈 **Per-Class Performance:**
- **Negative**: Precision=90.63%, Recall=88.95%, F1=89.78%
- **Neutral**: Precision=80.34%, Recall=85.16%, F1=82.68%
- **Positive**: Precision=92.27%, Recall=88.44%, F1=90.31%

### Performance by Class

| Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Negative  | 90.63%    | 88.95% | 89.78%   |
| Neutral   | 80.34%    | 85.16% | 82.68%   |
| Positive  | 92.27%    | 88.44% | 90.31%   |

### Confusion Matrix

The confusion matrix shows the model's performance across all classes:

```
              Predicted
              Negative  Neutral  Positive
Actual
Negative        1199       136        13
Neutral          113      1148        87
Positive          11       145      1193
```

#### Per-Class Accuracy

- **Negative class accuracy**: 88.95%
- **Neutral class accuracy**: 85.16%
- **Positive class accuracy**: 88.44%

## Model Inference

After training, you can use the `predict_sentiment()` function to analyze new financial text:

```python
text = "The company reported record profits in the last quarter."
result = predict_sentiment(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
print(f"Probabilities: {result['probabilities']}")
```

### Sample Predictions

| Text | Predicted Sentiment | Confidence |
|------|---------------------|------------|
| "Reliance Industries reported record profits with a 25% increase in quarterly earnings." | Positive | 95% |
| "The company's stock plummeted after disappointing earnings and lowered guidance." | Negative | 92% |
| "Infosys announced steady performance with results in line with market expectations." | Neutral | 88% |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- ProsusAI for the FinBERT model
- kdave for the Indian Financial News dataset

---

⭐ If this project helped you, please consider giving it a star!
