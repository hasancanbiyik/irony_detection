import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

####################################################################
''' (1) CONFIGURATION '''
####################################################################

# PATH to your saved fine-tuned model
MODEL_PATH = os.path.join(os.getcwd(), "saved_models/finetuned_0")

# LIST of datasets you want to test
# Format: ("path/to/csv", "name_of_dataset")
DATASETS_TO_TEST = [
    ("/Users/hasancan/Desktop/EXPERIMENT_RUNNER_4/sample_dataset_old_cleaned.csv", "sample_dataset_old_cleaned"),
    ("/Users/hasancan/Desktop/EXPERIMENT_RUNNER_4/sample_dataset_updated_cleaned.csv", "sample_dataset_updated_cleaned"),
]

# Column names in your new CSVs
TEXT_COLUMN = "text"   # The column containing the tweet/sentence
LABEL_COLUMN = "label" # The column containing 0 or 1 (Optional: set to None if you don't have labels yet)

# Hardware Settings
BATCH_SIZE = 8  # Keep this low (8) for your Mac M4
MAX_LEN = 512

####################################################################
''' (2) SETUP '''
####################################################################

# Detect Device (Mac M4 Optimization)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"[INFO] Using Apple MPS (Metal Performance Shaders) acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load Tokenizer and Model
print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval() # Set model to evaluation mode (turns off dropout/training layers)
except Exception as e:
    print(f"[ERROR] Could not load model. Did you finish training? Error: {e}")
    exit()

####################################################################
''' (3) HELPER CLASS '''
####################################################################

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

####################################################################
''' (4) INFERENCE LOOP '''
####################################################################

def run_inference(file_path, dataset_name):
    print(f"\n--- Processing: {dataset_name} ---")
    
    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}. Skipping.")
        return

    df = pd.read_csv(file_path)
    
    # Check columns
    if TEXT_COLUMN not in df.columns:
        print(f"[ERROR] Column '{TEXT_COLUMN}' not found in {file_path}. Skipping.")
        return

    # 2. Prepare Data Loader
    dataset = InferenceDataset(df[TEXT_COLUMN].to_numpy(), tokenizer, MAX_LEN)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    probabilities = []

    # 3. Predict Loop
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Predicting {dataset_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities (Softmax) and hard predictions (Argmax)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy().tolist())

    # 4. Save Results
    df['predicted_label'] = predictions
    # Optional: Save confidence scores for class 1 (Yes)
    df['confidence_score'] = [p[1] for p in probabilities] 

    output_filename = f"predictions_{dataset_name}.csv"
    df.to_csv(output_filename, index=False)
    print(f"[SUCCESS] Predictions saved to: {output_filename}")

    # 5. Calculate Metrics (Only if ground truth labels exist)
    if LABEL_COLUMN in df.columns:
        y_true = df[LABEL_COLUMN].to_numpy()
        y_pred = np.array(predictions)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"Metrics for {dataset_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")

####################################################################
''' (5) EXECUTION '''
####################################################################

if __name__ == "__main__":
    for csv_file, name in DATASETS_TO_TEST:
        run_inference(csv_file, name)
