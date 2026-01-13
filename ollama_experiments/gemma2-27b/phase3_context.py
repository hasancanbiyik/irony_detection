import ollama
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time

# --- CONFIGURATION ---
# We use Qwen 2.5 14B because it was your best performer.
MODEL_NAME = "gemma2:27b" 
TEMPERATURE = 0.0
LABEL_MAP = {0: "literal", 1: "sarcastic"}

# Replace this with your actual file name
CSV_FILE = "updated_dataset_with_context.csv"

# --- THE CONTEXT-AWARE PROMPT ---
# This structure forces the model to ground the "Text" in the "Context".
PROMPT_CONTEXT_ZEROSHOT = """You are an expert in pragmatics and conversational analysis.
Your task is to determine if the "Response" is sarcastic or literal, given the "Context".

Step 1: Read the Context to understand the situation.
Step 2: Read the Response.
Step 3: If the Response contradicts the context or mocks the situation, classify it as 'sarcastic'.
Step 4: If the Response is a sincere statement of fact or feeling, classify it as 'literal'.

Context: "{context}"
Response: "{text}"

Classification (return only 'sarcastic' or 'literal'):"""

def classify_with_context(context, text, model):
    # Inject both columns into the prompt
    full_prompt = PROMPT_CONTEXT_ZEROSHOT.format(context=context, text=text)
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={'temperature': TEMPERATURE}
        )
        content = response['message']['content'].strip().lower()
        
        # Robust parsing to catch "The response is sarcastic"
        if "non-sarcastic" in content or "literal" in content:
            return "literal"
        elif "sarcastic" in content:
            return "sarcastic"
        else:
            return "error"
    except Exception as e:
        print(f"Error: {e}")
        return "error"

def run_experiment():
    print(f"\n--- Starting Context-Aware Experiment with {MODEL_NAME} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_FILE)
        # Check required columns
        required_cols = ['text', 'label', 'context']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
            
        # Map numeric labels if necessary
        if df['label'].dtype != 'O': # If not already string
            df['true_label_str'] = df['label'].map(LABEL_MAP)
        else:
            df['true_label_str'] = df['label'].astype(str).str.lower()
            
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Run Inference
    predictions = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Context"):
        pred = classify_with_context(row['context'], row['text'], MODEL_NAME)
        predictions.append(pred)
        
    df['pred_context'] = predictions

    # 3. Filter Errors
    valid_df = df[df['pred_context'] != 'error']
    error_count = len(df) - len(valid_df)
    if error_count > 0:
        print(f"Warning: {error_count} errors during classification.")

    # 4. Save Results
    output_filename = "results_phase3_context.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nSaved detailed results to {output_filename}")

    # 5. Metrics
    print(f"\n--- Final Results ({len(valid_df)}/{len(df)} samples) ---")
    print(classification_report(valid_df['true_label_str'], valid_df['pred_context'], target_names=['literal', 'sarcastic']))
    print("Confusion Matrix:")
    print(confusion_matrix(valid_df['true_label_str'], valid_df['pred_context'], labels=['literal', 'sarcastic']))

if __name__ == "__main__":
    run_experiment()
