import ollama
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time

# --- CONFIGURATION ---
MODEL_NAME = "gemma2:9b" 
TEMPERATURE = 0.0
LABEL_MAP = {0: "literal", 1: "sarcastic"}

# --- DATASET SELECTION ---
# Un-comment the one you want to run
CSV_FILE = "sample_dataset_old_cleaned.csv"

# ==========================================
# PROMPT 1: The "SarcasmBench" Few-Shot (Baseline)
# ==========================================
# Based on Section 4.6 of the paper: 1 sarcastic example + 1 literal example.
# I chose a "tricky" literal example to help fix the precision error from Phase 1.

PROMPT_FEW_SHOT_PAPER = """This is a sarcasm classification task. Determine whether the following input text expresses sarcasm.
If it does, output 'sarcastic', otherwise, output 'literal'.
Return the label only without any other text.

Input: "Oh great, my car broke down again. I just love spending my weekends at the mechanic."
Output: sarcastic

Input: "I really love spending my weekends hiking in the mountains with my dog."
Output: literal

Input: "{text}"
Output:"""

# List of prompts to run in this batch
PROMPT_LIST = [
    {"name": "few_shot_baseline", "template": PROMPT_FEW_SHOT_PAPER},
]

def classify_text(text, model, prompt_template):
    full_prompt = prompt_template.format(text=text)
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={'temperature': TEMPERATURE}
        )
        content = response['message']['content'].strip().lower()
        
        if "non-sarcastic" in content or "literal" in content:
            return "literal"
        elif "sarcastic" in content:
            return "sarcastic"
        else:
            return "error"
    except Exception as e:
        print(f"Error: {e}")
        return "error"

def run_batch():
    # Load Data
    try:
        df = pd.read_csv(CSV_FILE)
        df['true_label_str'] = df['label'].map(LABEL_MAP)
    except Exception as e:
        print(f"Could not load {CSV_FILE}: {e}")
        return

    # Iterate through each prompt strategy
    for prompt_config in PROMPT_LIST:
        run_name = prompt_config["name"]
        template = prompt_config["template"]
        
        print(f"\n>>> Running Experiment: {run_name} ...")
        
        predictions = []
        start_time = time.time()
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            pred = classify_text(row['text'], MODEL_NAME, template)
            predictions.append(pred)
            
        duration = time.time() - start_time
        df[f'pred_{run_name}'] = predictions
        
        # Calculate Metrics for this specific run
        valid_df = df[df[f'pred_{run_name}'] != 'error']
        
        print(f"\n--- Results for {run_name} ({len(valid_df)}/{len(df)} valid) ---")
        print(classification_report(valid_df['true_label_str'], valid_df[f'pred_{run_name}'], target_names=['literal', 'sarcastic']))
        print("Confusion Matrix:")
        print(confusion_matrix(valid_df['true_label_str'], valid_df[f'pred_{run_name}'], labels=['literal', 'sarcastic']))
        
    # Save combined results
    output_file = "fewshot_old-data_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    run_batch()
