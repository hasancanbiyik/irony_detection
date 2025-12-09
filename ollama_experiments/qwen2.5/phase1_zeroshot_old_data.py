import ollama
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm  # pip install tqdm (for a progress bar)

# --- CONFIGURATION ---
MODEL_NAME = "qwen2.5:14b"
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

def classify_text(text, model, prompt_template, debug_errors=False):
    """
    Classifies a single string using a specific prompt template.
    """
    # Inject the text into the template
    full_prompt = prompt_template.format(text=text)
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': full_prompt}],
            options={
                'temperature': TEMPERATURE,
                'num_predict': 10  # Increase token limit to prevent truncation
            }
        )
        raw_content = response['message']['content'].strip()
        content = raw_content.lower()
        
        # IMPROVED PARSING LOGIC
        # Check for exact or close matches first (including truncated versions)
        if content in ["sarcastic", "sarcasm", "sarcaste"]:  # Added "sarcaste" for truncation
            return "sarcastic"
        elif content in ["literal", "non-sarcastic", "not sarcastic", "no", "litera"]:  # Added "litera" for truncation
            return "literal"
        
        # Then check for keywords in longer responses
        # Check for "not sarcastic" or "non-sarcastic" BEFORE checking for "sarcastic"
        if "not sarcastic" in content or "non-sarcastic" in content or "non sarcastic" in content:
            return "literal"
        elif "sarcastic" in content or "sarcasm" in content or "sarcaste" in content:  # Added "sarcaste"
            return "sarcastic"
        elif "literal" in content or "litera" in content:  # Added "litera"
            return "literal"
        else:
            # Log the problematic response for debugging
            if debug_errors:
                print(f"\n{'='*80}")
                print(f"UNPARSEABLE RESPONSE")
                print(f"Text: {text[:100]}...")
                print(f"Model response: '{raw_content}'")
                print(f"{'='*80}")
            return "error"
            
    except Exception as e:
        print(f"Exception error: {e}")
        return "error"

def run_experiment(csv_path, prompt_template, run_name="experiment", debug_first_n_errors=5):
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
    error_count = 0
    
    # tqdm gives you a nice progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        # Debug the first few errors to see what's happening
        debug = error_count < debug_first_n_errors
        pred = classify_text(row['text'], MODEL_NAME, prompt_template, debug_errors=debug)
        if pred == "error":
            error_count += 1
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
        # Show some examples of errors
        error_df = df[df['prediction'] == 'error']
        print("\nSample of texts that caused errors:")
        print(error_df[['text', 'prediction']].head(5))

    # 4. Save Results
    output_filename = f"zeroshot_old-data_{run_name}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Saved detailed results to {output_filename}")

    # 5. Print Metrics
    if len(valid_df) > 0:
        print(f"\nResults for: {run_name}")
        print(classification_report(valid_df['true_label_str'], valid_df['prediction'], target_names=['literal', 'sarcastic']))
        print("Confusion Matrix:")
        print(confusion_matrix(valid_df['true_label_str'], valid_df['prediction'], labels=['literal', 'sarcastic']))
    else:
        print("No valid predictions to evaluate!")

# --- EXECUTION ---
if __name__ == "__main__":
    # Create dummy CSVs for demonstration (Run this once if you don't have files yet)
    # pd.DataFrame({'text': ['I love queues.', 'It is sunny.'], 'label': [1, 0]}).to_csv('dataset_1.csv', index=False)
    
    # Run 1: Using the Paper's Prompt
    run_experiment("/Users/hasancan/Desktop/irony_detection/ollama_experiments/sample_dataset_old_cleaned.csv", PROMPT_PAPER, run_name="paper_prompt")
    
    # Run 2: Using Your Custom Prompt
    # run_experiment("dataset_1.csv", PROMPT_CUSTOM, run_name="custom_prompt")
