import os
import logging
import time
from datetime import datetime
from train import run_trainer

####################################################################
''' (0) SET EXPERIMENTAL DIRECTORY '''
####################################################################
# Automatically uses the current directory where you run the script
EXP_DIR = os.getcwd()

####################################################################
''' (1) CONFIGURE EXPERIMENT SETTINGS '''
####################################################################

# Use the specific Cardiff NLP model for Sentiment or Hate Speech
os.environ['MODEL_NAME'] = "cardiffnlp/twitter-roberta-base-sentiment" 
os.environ['EXP_DIR'] = EXP_DIR 
os.environ['SAVE_MODELS'] = 'True' 

# Logging note
os.environ['LOG'] = 'Fine-tuning Twitter RoBERTa on English dataset'

# File selection (Controlled by manifest.py)
# Make sure these numbers match the files you have (e.g., train_0.csv)
os.environ['START_FILENUM'] = "0"
os.environ['END_FILENUM'] = "1" 

# Hyperparameters
os.environ['NUM_EPOCHS'] = '5'
os.environ['LEARNING_RATE'] = '2e-5'
os.environ['WARMUP_STEPS'] = '100'
os.environ['BATCH_SIZE'] = '8'

# Training settings
os.environ['TRAIN_RESULTS_CSV'] = os.path.join(EXP_DIR, 'results_train.csv')
os.environ['EARLY_STOPPING_PATIENCE'] = "3"
os.environ['LAYERS_TO_FREEZE'] = '-1' 
os.environ['SPECIAL_TOKENS'] = '-1'

# Evaluation settings
os.environ['EVALUATION'] = 'True' 
os.environ['TEST_RESULTS_CSV'] = os.path.join(EXP_DIR, 'results_test.csv')
os.environ['LANGS'] = "['english']" 

####################################################################
''' (2) RUN LOOP '''
####################################################################

# --- CRITICAL FIX: Import manifest ONLY AFTER setting the variables above ---
from manifest import manifest 
# --------------------------------------------------------------------------

logging.basicConfig(format='[%(name)s] [%(levelname)s] %(asctime)s %(message)s', level=logging.INFO)

# Create notes file
with open(os.path.join(EXP_DIR, "notes.txt"), "a") as f:
    f.write(f"\n\n--- NEW RUN: {datetime.now()} ---\n")
    f.write(os.getenv('LOG') + '\n')
    for name, value in os.environ.items():
        if name in ['MODEL_NAME', 'NUM_EPOCHS', 'LEARNING_RATE']:
            f.write(f"{name}: {value}\n")

start = time.time()

# Check if manifest is empty
if not manifest:
    logging.warning("Manifest is empty. Check START_FILENUM and END_FILENUM in launch.py vs manifest.py")

for item in manifest:
    traindir = os.path.join(EXP_DIR, item['trainfile'])
    testdir = os.path.join(EXP_DIR, item['testfile'])
    model_output_dir = os.path.join(EXP_DIR, 'saved_models', item['model_name'])

    # Ensure model output dir exists
    os.makedirs(model_output_dir, exist_ok=True)

    if not os.path.exists(traindir) or not os.path.exists(testdir):
        logging.critical(f"Missing dataset files: {traindir} or {testdir}")
        continue

    logger = logging.getLogger(item['model_name'])
    
    try:
        model = run_trainer(traindir, testdir, model_output_dir, logger)
        del model
        # Clear GPU memory to prevent OOM in loop
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Failed processing {item['model_name']}: {str(e)}")

duration = time.time() - start
with open(os.path.join(EXP_DIR, "notes.txt"), "a") as f:
    f.write(f"DURATION: {duration:.2f} seconds\n")