import os
import sys
import logging
import ast
import numpy as np
import pandas as pd
import torch
from typing import Union
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    set_seed,
    EarlyStoppingCallback
)

def run_trainer(trainfile: str, 
                testfile: str, 
                output_dir: str, 
                logger: Union[logging.Logger, None], 
                seed: int = 42) -> AutoModelForSequenceClassification:
    
    # --- 1. Setup and Sanity Checks ---
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading data from: {trainfile} and {testfile}")
    
    for i in [trainfile, testfile]:
        if not os.path.exists(i):
            raise FileNotFoundError(f"File {i} does not exist.")

    set_seed(seed)
    
    # --- 2. Load Dataset & Tokenizer ---
    # Loading CSVs directly. 
    dataset = load_dataset("csv", data_files={"train": trainfile, "test": testfile})
    
    # Use environment variable or default to cardiffnlp
    model_name = os.getenv('MODEL_NAME', 'cardiffnlp/twitter-roberta-base-sentiment')
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    # Clean text function
    def preprocess_text(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(preprocess_text, batched=True)

    # --- 3. Prepare Model ---
    # Automatically determine number of labels from dataset
    unique_labels = sorted(list(set(dataset['train']['label'])))
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

    # --- 4. Metrics Calculation ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate core metrics
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

    # --- 5. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE', 3)))]
    )

    # --- 6. Train ---
    logger.info("Starting training...")
    trainer.train()

    # --- 7. Final Evaluation (The Fast Way) ---
    if os.getenv('EVALUATION', 'False') == 'True':
        logger.info("Running final evaluation on test set...")
        
        # This replaces the slow row-by-row loop
        predictions_output = trainer.predict(tokenized_datasets["test"])
        preds = np.argmax(predictions_output.predictions, axis=-1)
        labels = predictions_output.label_ids
        
        # Calculate Metrics
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        # --- FIXED TYPO BELOW: changed 'predictions' to 'preds' ---
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        
        # Handle Confusion Matrix 
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = -1, -1, -1, -1 
        
        # --- Save Results to CSV ---
        test_output_csv = os.getenv('TEST_RESULTS_CSV', 'results_test.csv')
        
        try:
            test_num = re.search(r'\d+', os.path.basename(trainfile)).group(0)
        except:
            test_num = "0"
            
        results_data = {
            'TEST_NUM': test_num,
            'LANG': 'english',
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

    # --- 8. Save Model ---
    if os.getenv('SAVE_MODELS', 'False') == 'True':
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    return model
