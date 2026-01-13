import os
import logging
import time
from datetime import datetime
# Since launch.py is in 'src', and train.py is also in 'src', this import works 
# if you run the script as a module or if they are in the python path.
from train import run_trainer

####################################################################
''' (0) SET EXPERIMENTAL DIRECTORY '''
####################################################################
# Assumes you run this from the root folder (EXPERIMENT_RUNNER_7)
EXP_DIR = os.getcwd()
os.environ['EXP_DIR'] = EXP_DIR 

####################################################################
''' (1) CONFIGURE EXPERIMENT SETTINGS '''
####################################################################

# Model Settings
os.environ['MODEL_NAME'] = "FacebookAI/xlm-roberta-base" 
os.environ['SAVE_MODELS'] = 'True' 
os.environ['LOG'] = 'Fine-tuning XLM-R with Context (English + Korean)'

# Hyperparameters
os.environ['NUM_EPOCHS'] = '4'
os.environ['LEARNING_RATE'] = '2e-5'
os.environ['WARMUP_STEPS'] = '100'

# CRITICAL CHANGE: Context requires longer sequences. 
# 128 cuts off the conversation history. 512 is the max for RoBERTa.
os.environ['MAX_LEN'] = '512'

# Batch Size: 32 is great for Cluster GPUs (V100/A100)
# If it hits Out-Of-Memory (OOM), lower this to 16 or 8.
os.environ['BATCH_SIZE'] = '32'       

# Training mechanics
os.environ['EARLY_STOPPING_PATIENCE'] = "3"
os.environ['LAYERS_TO_FREEZE'] = '0' 

####################################################################
''' (2) DEFINE EXPERIMENTS (THE MANIFEST) '''
####################################################################

manifest = [
    {
        # Context-Aware Experiment (English + Korean)
        'model_name': 'xlmr_context_en_ko',
        # These filenames match what you have in EXPERIMENT_RUNNER_7
        'trainfile': 'train.csv', 
        'testfile': 'val.csv'     
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
    for key in ['MODEL_NAME', 'NUM_EPOCHS', 'LEARNING_RATE', 'BATCH_SIZE', 'MAX_LEN']:
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
        model = run_trainer(traindir, testdir, model_output_dir, logger)
        
        # Memory Cleanup
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