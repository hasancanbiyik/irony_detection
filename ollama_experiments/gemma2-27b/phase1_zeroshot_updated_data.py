import ollama
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm  # pip install tqdm (for a progress bar)

# --- CONFIGURATION ---
MODEL_NAME = "gemma2:27b"
TEMPERATURE = 0.0
# Define the mapping for your binary labels
LABEL_MAP = {0: "literal", 1: "sarcastic"}

# --- PROMPT TEMPLATES ---
# 1. The Paper's Prompt (Baseline)
PROMPT_PAPER = """This is a sarcasm classification task. Determine whether the following input text expresses sarcasm.
If it does, output 'sarcastic', otherwise, output 'literal'.
Return the label only without any other text.

Input: "{text}"
Output:"""

def classify_text(text, model, prompt_template):
    """
    Classifies a single string using a specific prompt template.
    """
    # Inject the text into the template
    full_prompt = prompt_template.format(text=text)
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={'temperature': TEMPERATURE}
        )
        content = response['message']['content'].strip().lower()
        
        # Simple parser to catch "The answer is sarcastic" type responses
        if "non-sarcastic" in content or "literal" in content:
            return "literal"
        elif "sarcastic" in content:
            return "sarcastic"
        else:
            return "error"
            
    except Exception as e:
        print(f"Error: {e}")
        return "error"

def run_experiment(csv_path, prompt_template, run_name="experiment"):
    print(f"\n--- Starting Run: {run_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        # Ensure we have the right columns
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"CSV {csv_path} must have 'text' and 'label' columns.")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Run Inference
    predictions = []
    
    # tqdm gives you a nice progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        pred = classify_text(row['text'], MODEL_NAME, prompt_template)
        predictions.append(pred)
        
    df['prediction'] = predictions

    # 3. Process Labels for Metrics
    # Map the numeric ground truth (0/1) to string labels ("literal"/"sarcastic")
    df['true_label_str'] = df['label'].map(LABEL_MAP)
    
    # Filter out errors for metric calculation
    valid_df = df[df['prediction'] != 'error']
    error_count = len(df) - len(valid_df)
    
    if error_count > 0:
        print(f"Warning: {error_count} predictions were errors/unreadable.")

    # 4. Save Results
    output_filename = f"zeroshot_updated-data_{run_name}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Saved detailed results to {output_filename}")

    # 5. Print Metrics
    print(f"\nResults for: {run_name}")
    print(classification_report(valid_df['true_label_str'], valid_df['prediction'], target_names=['literal', 'sarcastic']))
    print("Confusion Matrix:")
    print(confusion_matrix(valid_df['true_label_str'], valid_df['prediction'], labels=['literal', 'sarcastic']))

# --- EXECUTION ---
if __name__ == "__main__":
    # Create dummy CSVs for demonstration (Run this once if you don't have files yet)
    # pd.DataFrame({'text': ['I love queues.', 'It is sunny.'], 'label': [1, 0]}).to_csv('dataset_1.csv', index=False)
    
    # Run 1: Using the Paper's Prompt
    run_experiment("/Users/hasancan/Desktop/irony_detection/ollama_experiments/sample_dataset_updated_cleaned.csv", PROMPT_PAPER, run_name="paper_prompt")
    
    # Run 2: Using Your Custom Prompt
    # run_experiment("dataset_1.csv", PROMPT_CUSTOM, run_name="custom_prompt")
