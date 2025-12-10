import os
import logging
import time
from datetime import datetime
from train import run_trainer

####################################################################
''' (0) SET EXPERIMENTAL DIRECTORY '''
####################################################################
EXP_DIR = os.getcwd()
os.environ['EXP_DIR'] = EXP_DIR 

####################################################################
''' (1) CONFIGURE EXPERIMENT SETTINGS '''
####################################################################

# Model Settings
os.environ['MODEL_NAME'] = "FacebookAI/xlm-roberta-base" 
os.environ['SAVE_MODELS'] = 'True' 
os.environ['LOG'] = 'Fine-tuning XLM-R on Multilingual Sarcasm Dataset'

# Hyperparameters
os.environ['NUM_EPOCHS'] = '4'
os.environ['LEARNING_RATE'] = '2e-5'
os.environ['WARMUP_STEPS'] = '100'
os.environ['BATCH_SIZE'] = '8'        # Increase to 16 if your GPU has >16GB VRAM
os.environ['MAX_LEN'] = '128'         # Standard for tweets/sentences

# Training mechanics
os.environ['EARLY_STOPPING_PATIENCE'] = "3"
os.environ['LAYERS_TO_FREEZE'] = '0'  # 0 = Fine-tune everything (Best for accuracy)

####################################################################
''' (2) DEFINE EXPERIMENTS (THE MANIFEST) '''
####################################################################
# This list defines exactly what will run.
# We run the BALANCED experiment first, then the FULL experiment.

manifest = [
    {
        # EXPERIMENT A: The "Fair Fight" (Small, balanced data)
        'model_name': 'xlm_roberta_balanced',
        'trainfile': 'train_balanced.csv',
        'testfile': 'final_val_gold.csv'  # We use the VALIDATION set to tune during training
    },
    {
        # EXPERIMENT B: The "Max Data" (Imbalanced, large data)
        'model_name': 'xlm_roberta_full',
        'trainfile': 'train_full.csv',
        'testfile': 'final_val_gold.csv'
    }
]

####################################################################
''' (3) MAIN TRAINING LOOP '''
####################################################################

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Launcher")

# Log run details to file
with open(os.path.join(EXP_DIR, "run_notes.txt"), "a") as f:
    f.write(f"\n\n--- NEW RUN: {datetime.now()} ---\n")
    for key in ['MODEL_NAME', 'NUM_EPOCHS', 'LEARNING_RATE', 'BATCH_SIZE']:
        f.write(f"{key}: {os.getenv(key)}\n")

start_time = time.time()

for item in manifest:
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {item['model_name']}")
    print(f"   Training Data: {item['trainfile']}")
    print(f"   Validation Data: {item['testfile']}")
    print(f"{'='*60}\n")

    traindir = os.path.join(EXP_DIR, item['trainfile'])
    testdir = os.path.join(EXP_DIR, item['testfile'])
    
    # We save each model in its own folder
    model_output_dir = os.path.join(EXP_DIR, 'saved_models', item['model_name'])
    os.makedirs(model_output_dir, exist_ok=True)

    # Check files exist
    if not os.path.exists(traindir) or not os.path.exists(testdir):
        logger.critical(f"❌ MISSING FILES: Check {traindir} or {testdir}")
        continue
    
    # Set the specific CSV path for this run's results
    os.environ['TRAIN_RESULTS_CSV'] = os.path.join(EXP_DIR, f"results_{item['model_name']}.csv")

    try:
        # Run the trainer
        # Note: This assumes your run_trainer function returns the trained model
        model = run_trainer(traindir, testdir, model_output_dir, logger)
        
        # Memory Cleanup (Crucial for running multiple experiments in one script)
        del model
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✅ FINISHED: {item['model_name']}\n")

    except Exception as e:
        logger.error(f"❌ CRASHED: {item['model_name']} failed with error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n🎉 ALL EXPERIMENTS COMPLETED in {(time.time() - start_time)/60:.2f} minutes.")