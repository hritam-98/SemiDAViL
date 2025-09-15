
# SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance

This repository contains a clean, complete PyTorch implementation of the paper **"SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation"**.

## Overview

SemiDAVIL is a framework for Semi-Supervised Domain Adaptation (SSDA) that improves semantic segmentation performance when adapting from a labeled source domain (e.g., synthetic data) to a target domain with very few labels (e.g., real-world scenes). It uniquely integrates guidance from Vision-Language Models (VLMs) to enhance semantic understanding and mitigate class confusion.



---

## Key Features

* **Language-Guided SSDA**: Leverages pre-trained CLIP models to create a semantic bridge between domains.
* **Dense Language Guidance (DLG)**: A novel fusion module for deep, symmetric integration of visual and textual features.
* **Dynamic Cross-Entropy (DyCE) Loss**: An adaptive loss function that combats class imbalance by focusing on the hardest examples and re-weighting tail classes.
* **Student-Teacher Framework**: Employs consistency regularization on unlabeled target data for robust learning.

---

## File Structure


.
├── data/
│   ├── datasets.py     \# PyTorch Dataset classes for GTA5 & Cityscapes
│   └── loader.py       \# Factory function to create DataLoaders
├── models/
│   ├── dlg\_module.py     \# Dense Language Guidance (DLG) module
│   └── semidavil.py      \# Main SemiDAVIL model architecture
├── config.py             \# Central configuration for all hyperparameters and paths
├── losses.py             \# Custom loss functions (DyCE, Consistency)
├── train.py              \# Main training script
├── utils.py              \# Helper functions (EMA update)
├── requirements.txt      \# Project dependencies
└── README.md             \# This file




## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/SemiDAVIL.git](https://github.com/your-username/SemiDAVIL.git)
    cd SemiDAVIL
    ```

2.  **Create a Conda/Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    
    # On Linux/macOS
    source venv/bin/activate
    
    # On Windows
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Datasets:**
    * Download the **GTA5** dataset (images and labels).
    * Download the **Cityscapes** dataset (leftImg8bit and gtFine).

---

## How to Run

1.  **Configure Paths:** Open `config.py` and update the `GTA5_DATA_PATH` and `CITYSCAPES_DATA_PATH` variables to point to the root directories of your datasets.

2.  **Adjust Hyperparameters (Optional):** Modify settings like `BATCH_SIZE`, `LEARNING_RATE`, or `NUM_LABELED_TARGET_SAMPLES` in `config.py` to match your desired experimental setup.

3.  **Start Training:**
    ```bash
    python train.py
    ```
    The script will initialize the models, load the datasets, and begin the training loop. Progress will be logged to the console, and the final trained model will be saved in the `./checkpoints` directory.

---

## Citation

If this implementation is useful for your research, please cite the original paper:

```bibtex
@inproceedings{basak2025semidavil,
  title={SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation},
  author={Basak, Hritam and Yin, Zhaozheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
````

```
```


