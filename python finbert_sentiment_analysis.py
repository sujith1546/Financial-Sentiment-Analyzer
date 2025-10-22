"""
Financial Sentiment Analysis using FinBERT
Fine-tuned on Indian Financial News Dataset
Google Colab Ready - Complete Implementation
"""

# ============================================
# STEP 1: SETUP & INSTALLATION
# ============================================

# Install required packages
!pip install datasets transformers torch pandas numpy scikit-learn matplotlib seaborn
!pip install accelerate -U

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================
# STEP 2: LOAD AND EXPLORE DATASET
# ============================================

print("\n" + "="*60)
print("LOADING INDIAN FINANCIAL NEWS DATASET")
print("="*60)

# Load the dataset from Hugging Face
dataset = load_dataset("kdave/Indian_Financial_News")
print(f"\nDataset loaded successfully!")
print(f"Dataset structure: {dataset}")

# Convert to pandas for exploration
df = dataset['train'].to_pandas()
print(f"\nTotal samples: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# The dataset has columns: URL, Content, Summary, Sentiment
# Rename for easier handling (use lowercase for consistency)
df = df.rename(columns={'Sentiment': 'sentiment', 'Content': 'content', 'Summary': 'summary'})

# Display sentiment distribution
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# Visualize sentiment distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'gray', 'red'])
plt.title('Sentiment Percentage')
plt.ylabel('')
plt.tight_layout()
plt.show()


# ============================================
# STEP 3: DATA PREPROCESSING
# ============================================

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Clean the data
df = df.dropna(subset=['content', 'sentiment'])

# Normalize sentiment labels (they might be 'Positive', 'Negative', 'Neutral' with capital letters)
df['sentiment'] = df['sentiment'].str.lower().str.strip()

# Map sentiment labels to numerical values
sentiment_mapping = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}

# Reverse mapping for interpretation
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

df['label'] = df['sentiment'].map(sentiment_mapping)

# Remove any rows with unmapped labels
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"\nCleaned dataset size: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess financial text"""
    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['text'] = df['content'].apply(preprocess_text)

# Check text length distribution
df['text_length'] = df['text'].str.len()
print(f"\nText length statistics:")
print(df['text_length'].describe())

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['text_length'], bins=50, edgecolor='black')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths')

plt.subplot(1, 2, 2)
plt.boxplot(df['text_length'])
plt.ylabel('Text Length')
plt.title('Text Length Boxplot')
plt.tight_layout()
plt.show()


# ============================================
# STEP 4: TRAIN-VALIDATION-TEST SPLIT
# ============================================

from sklearn.model_selection import train_test_split

print("\n" + "="*60)
print("SPLITTING DATASET")
print("="*60)

# Split: 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(
    df, 
    test_size=0.3, 
    random_state=42, 
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df['label']
)

print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True))

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print("\nDataset splits created successfully!")


# ============================================
# STEP 5: LOAD FINBERT MODEL & TOKENIZER
# ============================================

print("\n" + "="*60)
print("LOADING FINBERT MODEL")
print("="*60)

# Load FinBERT tokenizer and model
model_name = "ProsusAI/finbert"

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

model.to(device)
print(f"Model loaded successfully on {device}!")


# ============================================
# STEP 6: TOKENIZATION
# ============================================

print("\n" + "="*60)
print("TOKENIZING DATASETS")
print("="*60)

# Tokenization function
def tokenize_function(examples):
    """Tokenize text with padding and truncation"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

# Tokenize all datasets
print("Tokenizing training set...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

print("Tokenizing validation set...")
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

print("Tokenizing test set...")
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

print("Tokenization complete!")

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


# ============================================
# STEP 7: DEFINE METRICS
# ============================================

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================
# STEP 8: TRAINING CONFIGURATION
# ============================================

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    report_to='none',
    seed=42
)

print("Training Configuration:")
print(f"- Epochs: {training_args.num_train_epochs}")
print(f"- Batch size: {training_args.per_device_train_batch_size}")
print(f"- Learning rate: {training_args.learning_rate}")
print(f"- Mixed precision: {training_args.fp16}")


# ============================================
# STEP 9: INITIALIZE TRAINER
# ============================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nTrainer initialized successfully!")


# ============================================
# STEP 10: FINE-TUNE MODEL
# ============================================

print("\n" + "="*60)
print("STARTING FINE-TUNING")
print("="*60)

# Start training
start_time = datetime.now()
print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

train_result = trainer.train()

end_time = datetime.now()
training_duration = end_time - start_time
print(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Training duration: {training_duration}")

# Save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("\n" + "="*60)
print("TRAINING METRICS")
print("="*60)
for key, value in metrics.items():
    print(f"{key}: {value}")


# ============================================
# STEP 11: EVALUATE ON VALIDATION SET
# ============================================

print("\n" + "="*60)
print("VALIDATION SET EVALUATION")
print("="*60)

val_results = trainer.evaluate(tokenized_val)
print("\nValidation Results:")
for key, value in val_results.items():
    print(f"{key}: {value:.4f}")


# ============================================
# STEP 12: EVALUATE ON TEST SET
# ============================================

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

test_results = trainer.evaluate(tokenized_test)
print("\nTest Results:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")

# Get predictions
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# Display final accuracy
test_accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n🎯 Final Test Accuracy: {test_accuracy*100:.2f}%")


# ============================================
# STEP 13: CONFUSION MATRIX
# ============================================

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Negative', 'Neutral', 'Positive'],
    yticklabels=['Negative', 'Neutral', 'Positive']
)
plt.title('Confusion Matrix - Financial Sentiment Analysis', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_accuracy):
    print(f"{id2label[i].capitalize()} class accuracy: {acc*100:.2f}%")


# ============================================
# STEP 14: CLASSIFICATION REPORT
# ============================================

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)

# Generate classification report
report = classification_report(
    true_labels, 
    pred_labels,
    target_names=['Negative', 'Neutral', 'Positive'],
    digits=4
)
print("\n" + report)

# Get classification report as dictionary
report_dict = classification_report(
    true_labels, 
    pred_labels,
    target_names=['Negative', 'Neutral', 'Positive'],
    output_dict=True
)

# Visualize classification metrics
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :-1]
metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('Performance Metrics by Sentiment Class', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment Class', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# STEP 15: ERROR ANALYSIS
# ============================================

print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Find misclassified examples
misclassified_indices = np.where(pred_labels != true_labels)[0]
print(f"\nTotal misclassified samples: {len(misclassified_indices)}")
print(f"Misclassification rate: {len(misclassified_indices)/len(true_labels)*100:.2f}%")

# Show some misclassified examples
print("\nSample Misclassified Examples:")
print("-" * 80)

for i, idx in enumerate(misclassified_indices[:5]):
    text = test_df.iloc[idx]['text'][:200]  # First 200 chars
    true_sent = id2label[true_labels[idx]]
    pred_sent = id2label[pred_labels[idx]]
    
    print(f"\nExample {i+1}:")
    print(f"Text: {text}...")
    print(f"True Sentiment: {true_sent}")
    print(f"Predicted Sentiment: {pred_sent}")
    print("-" * 80)


# ============================================
# STEP 16: SAVE THE MODEL
# ============================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save the fine-tuned model
model_save_path = './finbert_indian_finance'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to: {model_save_path}")


# ============================================
# STEP 17: INFERENCE FUNCTION
# ============================================

def predict_sentiment(text):
    """
    Predict sentiment for a given financial text
    
    Args:
        text (str): Financial news text
        
    Returns:
        dict: Predicted sentiment and confidence scores
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    # Get all probabilities
    probs = {
        'negative': predictions[0][0].item(),
        'neutral': predictions[0][1].item(),
        'positive': predictions[0][2].item()
    }
    
    return {
        'sentiment': id2label[predicted_class],
        'confidence': confidence,
        'probabilities': probs
    }


# ============================================
# STEP 18: TEST PREDICTIONS
# ============================================

print("\n" + "="*60)
print("TESTING PREDICTIONS ON NEW EXAMPLES")
print("="*60)

# Test examples
test_examples = [
    "Reliance Industries reported record profits with a 25% increase in quarterly earnings.",
    "The company's stock plummeted after disappointing earnings and lowered guidance.",
    "Infosys announced steady performance with results in line with market expectations.",
    "HDFC Bank's merger with HDFC Ltd creates India's largest private sector bank.",
    "Adani Group faces significant challenges amid regulatory scrutiny and market concerns."
]

for i, example in enumerate(test_examples, 1):
    result = predict_sentiment(example)
    
    print(f"\nExample {i}:")
    print(f"Text: {example}")
    print(f"Predicted Sentiment: {result['sentiment'].upper()}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("Probabilities:")
    for sent, prob in result['probabilities'].items():
        print(f"  {sent.capitalize()}: {prob*100:.2f}%")
    print("-" * 80)


# ============================================
# STEP 19: MODEL PERFORMANCE SUMMARY
# ============================================

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

summary = f"""
📊 FINAL MODEL PERFORMANCE METRICS

Dataset: Indian Financial News (26,000+ articles)
Model: FinBERT (Fine-tuned)
Training Samples: {len(train_df)}
Validation Samples: {len(val_df)}
Test Samples: {len(test_df)}

🎯 Test Set Results:
- Accuracy: {test_accuracy*100:.2f}%
- Precision: {report_dict['weighted avg']['precision']*100:.2f}%
- Recall: {report_dict['weighted avg']['recall']*100:.2f}%
- F1-Score: {report_dict['weighted avg']['f1-score']*100:.2f}%

📈 Per-Class Performance:
- Negative: Precision={report_dict['Negative']['precision']*100:.2f}%, Recall={report_dict['Negative']['recall']*100:.2f}%, F1={report_dict['Negative']['f1-score']*100:.2f}%
- Neutral:  Precision={report_dict['Neutral']['precision']*100:.2f}%, Recall={report_dict['Neutral']['recall']*100:.2f}%, F1={report_dict['Neutral']['f1-score']*100:.2f}%
- Positive: Precision={report_dict['Positive']['precision']*100:.2f}%, Recall={report_dict['Positive']['recall']*100:.2f}%, F1={report_dict['Positive']['f1-score']*100:.2f}%

⏱️ Training Time: {training_duration}

💾 Model saved at: {model_save_path}

✅ Project completed successfully!
"""

print(summary)


# ============================================
# STEP 20: OPTIONAL - DOWNLOAD MODEL
# ============================================

print("\n" + "="*60)
print("DOWNLOAD MODEL (Optional)")
print("="*60)

print("""
To download your trained model from Google Colab:

1. Zip the model folder:
   !zip -r finbert_indian_finance.zip ./finbert_indian_finance

2. Download using files module:
   from google.colab import files
   files.download('finbert_indian_finance.zip')

3. Or mount Google Drive and copy:
   from google.colab import drive
   drive.mount('/content/drive')
   !cp -r ./finbert_indian_finance /content/drive/MyDrive/
""")

print("\n🎉 PROJECT COMPLETE! 🎉")
print("\nYour FinBERT model is now ready for financial sentiment analysis!")
