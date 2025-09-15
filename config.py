"""
Central configuration file for the SemiDAVIL project.
All hyperparameters, paths, and model settings are defined here.
"""
import torch

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ITERATIONS = 40000  # Total training iterations
BATCH_SIZE = 4          # Adjust based on your GPU memory
NUM_WORKERS = 4         # Number of workers for data loading
LEARNING_RATE = 1e-4    # Initial learning rate
WEIGHT_DECAY = 2e-4     # Weight decay for the optimizer
MOMENTUM = 0.9          # Momentum for the optimizer
CHECKPOINT_DIR = "./checkpoints" # Directory to save model checkpoints

# --- Dataset Paths ---
# IMPORTANT: Update these paths to your local dataset directories.
GTA5_DATA_PATH = "/path/to/your/gta5"
CITYSCAPES_DATA_PATH = "/path/to/your/cityscapes"
NUM_LABELED_TARGET_SAMPLES = 100 # Number of labeled Cityscapes samples (e.g., 100, 200, 500)
NUM_CLASSES = 19 # 19 classes for GTA5 -> Cityscapes adaptation

# --- Model Configuration ---
VLM_MODEL_NAME = "ViT-B/16"      # Vision-Language Model backbone from CLIP
CAPTION_MODEL_NAME = "Salesforce/blip2-opt-2.7b" # Off-the-shelf captioning model

# --- Teacher-Student Framework ---
EMA_ALPHA = 0.999 # Exponential Moving Average decay for teacher model updates

# --- Loss Function Parameters ---
# Consistency Loss
PSEUDO_LABEL_THRESHOLD = 0.95 # Confidence threshold for pseudo-labels

# Dynamic Cross-Entropy (DyCE) Loss
DYCE_HARD_PERCENTAGE = 0.2 # Percentage of hardest samples to mine (h% in the paper)
DYCE_OMEGA = 0.5           # Weight-balancing factor (ω in the paper)

