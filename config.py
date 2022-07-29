import torch

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DATA
DATA_DIR = ""
AUGMENT = True

# TRAINING
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
L1_LAMBDA = 100
