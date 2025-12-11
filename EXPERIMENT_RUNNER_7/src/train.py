import os
import sys
import logging
import ast
import numpy as np
import pandas as pd
import torch
import re
from typing import Union
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# NEW IMPORTS FOR CUSTOM DATASET
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback
)

# --- 1. CUSTOM DATASET CLASS (HANDLES CONTEXT + RESPONSE) ---
class SarcasmContextDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=512):
        # Handle filename input
        if isinstance(filename, list): filename = filename[0]
            
        # Load CSV
        self.df = pd.read_csv(filename)
        
        # Robust cleanup: Drop rows where any critical info is missing
        # We need 'context', 'text' (response), and 'label'
        self.df = self.df.dropna(subset=['context', 'text', 'label'])
        self.df['label'] = self.df['label'].astype(int)
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Store unique labels for model config
        self.unique_labels = sorted(list(set(self.df['label'])))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Sequence A: Context
        # Sequence B: Response (Text)
        context = str(row['context'])
        response = str(row['text'])
        label = int(row['label'])

        # TOKENIZATION STRATEGY:
        # We pass (context, response) as a pair.
        # truncation="only_first": If total length > 512, truncate the CONTEXT (Seq A),
        # preserving the RESPONSE (Seq B) completely.
        encoding = self.tokenizer(
            context,
            response,
            truncation="only_first", 
            max_length=self.max_len,
            padding="max_length", # Ensures uniform tensor shape
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def run_trainer(trainfile: str, 
                testfile: str, 
                output_dir: str, 
                logger: Union[logging.Logger, None], 
                seed: int = 42) -> AutoModelForSequenceClassification:
    
    # --- 2. Setup and Sanity Checks ---
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading data from: {trainfile} and {testfile}")
    
    for i in [trainfile, testfile]:
        if not os.path.exists(i):
            raise FileNotFoundError(f"File {i} does not exist.")

    set_seed(seed)
    
    # --- 3. Load Tokenizer ---
    # Default to XLM-R if not specified, or use env variable
    model_name = os.getenv('MODEL_NAME', 'FacebookAI/xlm-roberta-base')
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    logger.info("Initializing Context-Aware Datasets...")
    # Initialize the custom dataset (No separate 'map' step needed anymore)
    train_dataset = SarcasmContextDataset(trainfile, tokenizer, max_len=512)
    test_dataset = SarcasmContextDataset(testfile, tokenizer, max_len=512)

    # --- 4. Prepare Model ---
    unique_labels = train_dataset.unique_labels
    num_labels = len(unique_labels)
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}

    logger.info(f"Detected {num_labels} labels: {unique_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )

    # --- 5. Metrics Calculation ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        prec = precision_score(labels, predictions, average='macro', zero_division=0)
        rec = recall_score(labels, predictions, average='macro', zero_division=0)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec
        }

    # --- 6. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=float(os.getenv('NUM_EPOCHS', 3)),
        learning_rate=float(os.getenv('LEARNING_RATE', 2e-5)),
        per_device_train_batch_size=int(os.getenv('BATCH_SIZE', 8)),
        per_device_eval_batch_size=int(os.getenv('BATCH_SIZE', 8)),
        warmup_steps=int(os.getenv('WARMUP_STEPS', 0)),
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE', 3)))]
    )

    # --- 7. Train ---
    logger.info("Starting training...")
    trainer.train()

    # --- 8. Final Evaluation ---
    if os.getenv('EVALUATION', 'False') == 'True':
        logger.info("Running final evaluation on test set...")
        
        # Use the Trainer to predict on the custom test dataset
        predictions_output = trainer.predict(test_dataset)
        preds = np.argmax(predictions_output.predictions, axis=-1)
        labels = predictions_output.label_ids
        
        # Calculate Metrics
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        
        # Confusion Matrix 
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = -1, -1, -1, -1 
        
        # Save Results to CSV
        test_output_csv = os.getenv('TEST_RESULTS_CSV', 'results_test.csv')
        
        try:
            test_num = re.search(r'\d+', os.path.basename(trainfile)).group(0)
        except:
            test_num = "0"
            
        results_data = {
            'TEST_NUM': test_num,
            'LANG': 'context_aware', # Marking this as context run
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        
        df_new = pd.DataFrame([results_data])
        
        if not os.path.exists(test_output_csv):
            df_new.to_csv(test_output_csv, index=False)
        else:
            df_new.to_csv(test_output_csv, mode='a', header=False, index=False)

        logger.info(f"Results saved to {test_output_csv}")

    # --- 9. Save Model ---
    if os.getenv('SAVE_MODELS', 'False') == 'True':
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    return model