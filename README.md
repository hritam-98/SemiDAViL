# SemiDAViL

Code will be updated soon. Thanks for your patience :)

contact: hbasak@cs.stonybrook.edu

SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance
This repository contains the official PyTorch implementation for the paper: "SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation".

This project provides a novel framework for Semi-Supervised Domain Adaptation (SSDA) in semantic segmentation. It leverages the rich semantic knowledge from pre-trained Vision-Language Models (VLMs) like CLIP to bridge the domain gap between a labeled source domain (e.g., synthetic data) and a target domain with limited labeled samples (e.g., real-world data).

Key Contributions
Language-Guided SSDA Framework: The first framework to integrate VLM-based language guidance into the SSDA setting for semantic segmentation, enhancing semantic understanding and reducing class confusion.

Dense Language Guidance (DLG): A novel attention-based fusion module that deeply integrates visual features with dense language embeddings for a robust multimodal representation.

Dynamic Cross-Entropy (DyCE) Loss: An adaptive, class-balancing loss function that dynamically re-weights under-represented classes within each batch to tackle severe class imbalance, particularly for tail classes.

Architecture Overview
The SemiDAVIL framework is built on a student-teacher architecture that promotes learning through consistency regularization on unlabeled target data.

VL Initialization: The vision encoders for both the student and teacher networks are initialized with weights from a pre-trained CLIP model. The CLIP language encoder is used to generate text embeddings and its weights are kept frozen.

Captioning & Language Features: An off-the-shelf captioning model generates descriptive text for input images. These captions are then fed into the frozen language encoder to produce dense semantic feature vectors.

Dense Language Guidance (DLG): The visual features from the vision encoder and the language features are fused using the DLG module, which employs a cross-attention mechanism to create a powerful, language-aware feature map.

Consistency Training: The student model is trained on both labeled data (using the DyCE loss) and unlabeled data. The teacher model's predictions on unlabeled data serve as pseudo-labels to enforce consistency, guiding the student. The teacher's weights are updated via an Exponential Moving Average (EMA) of the student's weights.

File Structure
.
├── models/
│   ├── dlg_module.py     # Implementation of the Dense Language Guidance module
│   └── semidavil.py      # Main SemiDAVIL model architecture
├── config.py             # All hyperparameters and configuration settings
├── losses.py             # Custom loss functions (DyCE Loss, Consistency Loss)
├── train.py              # Main script for training the model
├── utils.py              # Helper functions (e.g., EMA update)
└── README.md             # This file

Setup and Installation
Clone the repository:

git clone [https://github.com/your-username/SemiDAVIL.git](https://github.com/your-username/SemiDAVIL.git)
cd SemiDAVIL

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Usage
Configure Paths: Open config.py and update the dataset paths (GTA5_DATA_PATH, CITYSCAPES_DATA_PATH, etc.) to point to your data directories.

Start Training: Run the main training script.

python train.py

The script will initialize the models, create dummy dataloaders for demonstration, and begin the training loop. Progress will be logged to the console, and the final model checkpoint will be saved in the ./checkpoints directory.

Note: The current train.py uses dummy data. You will need to implement a proper data loader for your datasets (e.g., GTA5, Cityscapes) to replicate the paper's results.

Configuration
All major hyperparameters can be tuned in config.py:

LEARNING_RATE, WEIGHT_DECAY: Optimizer settings.

NUM_ITERATIONS: Total number of training iterations.

EMA_ALPHA: Momentum for the teacher model update.

PSEUDO_LABEL_THRESHOLD: Confidence threshold for consistency training.

DYCE_OMEGA, DYCE_HARD_PERCENTAGE: Parameters for the DyCE loss.

VLM_MODEL_NAME, CAPTION_MODEL_NAME: Names of the pre-trained models to use.

Citation
If you find this work useful for your research, please consider citing the original paper:

@inproceedings{basak2025semidavil,
  title={SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation},
  author={Basak, Hritam and Yin, Zhaozheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
