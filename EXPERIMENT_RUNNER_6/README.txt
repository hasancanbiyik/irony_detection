================================================================================
MULTILINGUAL SARCASM DETECTION EXPERIMENT
Fine-tuning XLM-RoBERTa on English, German, Arabic, Chinese, and Korean
================================================================================

1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project explores the capabilities of multilingual Large Language Models (LLMs) 
in detecting sarcasm across typologically diverse languages. Specifically, we 
investigate whether "More Data" (Massive Multilingual Training) or "Balanced Data" 
(Equal representation per language) yields better performance, particularly for 
low-resource languages like German.

Model Architecture: XLM-RoBERTa Base (FacebookAI/xlm-roberta-base)
Task: Binary Classification (0 = Literal, 1 = Sarcastic)

2. DATASET & LANGUAGES
--------------------------------------------------------------------------------
The dataset is an aggregation of 5 distinct language sources, standardized into 
a unified format (text, label).

- English (EN): ~3,400 examples
- Arabic (AR):  ~3,300 examples
- Chinese (ZH): ~5,200 examples (Mandarin)
- Korean (KO):  ~10,400 examples
- German (DE):  ~210 examples (Low-Resource)

3. EXPERIMENTAL DESIGN
--------------------------------------------------------------------------------
To ensure scientific rigor, we created fixed "Gold Standard" evaluation sets that 
never touch the training process. We then compare two training strategies:

[A] THE "GOLD" EVALUATION SETS (Fixed)
    - final_test_gold.csv: The Final Exam. Contains exactly 40 examples per 
      language (20 Sarcastic, 20 Literal). Total: 200 rows.
    - final_val_gold.csv: The Validation Set. Used for tuning/early stopping. 
      Contains exactly 40 examples per language. Total: 200 rows.

[B] TRAINING STRATEGY 1: "BALANCED" (The Fair Fight)
    - File: train_balanced.csv (~670 rows)
    - Logic: Downsamples every language to match the size of the smallest 
      dataset (German).
    - Goal: Test if the model is biased by language frequency.

[C] TRAINING STRATEGY 2: "FULL" (Max Data)
    - File: train_full.csv (~22,000+ rows)
    - Logic: Uses all remaining available data.
    - Goal: Test if Cross-Lingual Transfer allows the model to learn sarcasm 
      concepts from Chinese/Arabic and apply them to German.

4. REPRODUCIBILITY STEPS
--------------------------------------------------------------------------------
Step 1: Environment Setup
    Install dependencies:
    pip install torch transformers pandas scikit-learn

Step 2: Data Preparation
    Run the master splitter script to generate the 4 key CSV files:
    python master_splitter.py
    (Outputs: final_test_gold.csv, final_val_gold.csv, train_balanced.csv, train_full.csv)

Step 3: Training
    Run the launcher script. It automatically executes both experiments sequentially:
    python launch.py
    
    Experiment 1: Fine-tunes on `train_balanced.csv` -> Saves to /saved_models/xlm_roberta_balanced
    Experiment 2: Fine-tunes on `train_full.csv`     -> Saves to /saved_models/xlm_roberta_full

Step 4: Evaluation
    Compare the accuracy of both models on `final_test_gold.csv`.

5. HYPERPARAMETERS
--------------------------------------------------------------------------------
- Epochs: 4
- Batch Size: 8
- Learning Rate: 2e-5
- Max Sequence Length: 128
- Optimizer: AdamW
- Early Stopping Patience: 3 epochs

6. FILE STRUCTURE
--------------------------------------------------------------------------------
/data/                   -> Raw CSV source files
launch.py                -> Main execution script
train.py                 -> Training logic (HuggingFace Trainer)
final_test_gold.csv      -> HOLD-OUT TEST SET (Do not use for training)
final_val_gold.csv       -> VALIDATION SET
train_balanced.csv       -> Training Set A
train_full.csv           -> Training Set B
results_*.csv            -> Logs of loss/accuracy per epoch

================================================================================
