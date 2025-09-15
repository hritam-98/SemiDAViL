# config.py
# This file contains all hyperparameters and configuration settings for the SemiDAVIL project.
# Centralizing configuration makes it easy to tune the model and experiment with different settings.

import torch

# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 4  # Adjust based on your GPU memory
NUM_ITERATIONS = 40000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-4
LR_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

# --- Model Configuration ---
# Vision-Language Pre-trained Model (CLIP)
VLM_MODEL_NAME = "ViT-B/16"

# Off-the-shelf Captioning Model (BLIP-2)
# Using a smaller, more accessible version for demonstration.
# The paper uses BLIP-2, which can be swapped in if resources permit.
CAPTION_MODEL_NAME = "Salesforce/blip-base"


# --- SSDA (Semi-Supervised Domain Adaptation) Configuration ---
# EMA (Exponential Moving Average) momentum for updating the teacher model
EMA_ALPHA = 0.999

# Confidence threshold for pseudo-labels in Consistency Training
PSEUDO_LABEL_THRESHOLD = 0.95

# --- Dynamic Cross-Entropy (DyCE) Loss Configuration ---
# The 'omega' hyperparameter for the DyCE loss, balancing instance and class weighting.
# Described in Section 3.4 of the paper.
DYCE_OMEGA = 0.5

# The percentage of the hardest samples to mine from each batch for DyCE loss calculation.
DYCE_HARD_PERCENTAGE = 0.20 # Using 20% as a starting point

# --- Dataset Configuration ---
# Image dimensions as specified in Section 4.2 of the paper
SOURCE_IMG_SIZE = (1280, 760)
TARGET_IMG_SIZE = (1024, 512)
CROP_SIZE = (512, 512)

# Number of classes for the segmentation task (e.g., Cityscapes has 19 classes)
NUM_CLASSES = 19

# Paths to datasets (replace with your actual paths)
GTA5_DATA_PATH = "./data/gta5"
SYNTHIA_DATA_PATH = "./data/synthia"
CITYSCAPES_DATA_PATH = "./data/cityscapes"

# --- Logging and Checkpoints ---
LOG_INTERVAL = 100
CHECKPOINT_DIR = "./checkpoints"
