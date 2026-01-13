import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# CONFIG
MODEL_PATH = "./saved_models/xlmr_context_en_ko"  # Where your trained model is
TEST_FILE = "test.csv"

def evaluate_english():
    print(f"Loading English Test Set: {TEST_FILE}...")
    try:
        df = pd.read_csv(TEST_FILE)
    except:
        print("❌ File not found.")
        return

    # Load Model & Tokenizer
    print(f"Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Running Predictions...")
    preds = []
    
    # Loop through rows (since it's small, simple loop is fine)
    for index, row in df.iterrows():
        context = str(row['context'])
        text = str(row['text'])
        
        # Tokenize (using the same Context strategy)
        inputs = tokenizer(
            context, 
            text, 
            truncation="only_first", 
            max_length=512, 
            padding="max_length", 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            preds.append(pred)

    # Metrics
    labels = df['label'].astype(int).tolist()
    acc = accuracy_score(labels, preds)
    
    print("\n" + "="*40)
    print(f"ENGLISH TEST RESULTS (N={len(df)})")
    print("="*40)
    print(f"Accuracy: {acc:.2%}")
    print("\nDetailed Report:")
    print(classification_report(labels, preds, target_names=["Literal", "Sarcastic"]))
    
    # Save predictions to see what it got right/wrong
    df['prediction'] = preds
    df['correct'] = df['label'] == df['prediction']
    df.to_csv("english_predictions.csv", index=False)
    print("Saved detailed predictions to 'english_predictions.csv'")

if __name__ == "__main__":
    evaluate_english()