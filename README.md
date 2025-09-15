# SemiDAViL

Code will be updated soon. Thanks for your patience :)

contact: hbasak@cs.stonybrook.edu

# SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance

This repository contains a clean, complete PyTorch implementation of the paper:  
**"SemiDAVIL: Semi-supervised Domain Adaptation with Vision-Language Guidance for Semantic Segmentation".**

---

## 📌 Overview
SemiDAVIL is a framework for **Semi-Supervised Domain Adaptation (SSDA)** that improves semantic segmentation performance when adapting from a labeled source domain (e.g., synthetic data) to a target domain with very few labels (e.g., real-world scenes).  

It uniquely integrates **Vision-Language Models (VLMs)** to enhance semantic understanding and mitigate class confusion.

---

## 🚀 Key Features
- **Language-Guided SSDA**: Leverages pre-trained CLIP models to create a semantic bridge between domains.  
- **Dense Language Guidance (DLG)**: A novel fusion module for deep, symmetric integration of visual and textual features.  
- **Dynamic Cross-Entropy (DyCE) Loss**: An adaptive loss function that combats class imbalance by focusing on the hardest examples and re-weighting tail classes.  
- **Student-Teacher Framework**: Employs consistency regularization on unlabeled target data for robust learning.  

---

## 📂 File Structure
├── data/
│ ├── datasets.py # PyTorch Dataset classes for GTA5 & Cityscapes
│ └── loader.py # Factory function to create DataLoaders
├── models/
│ ├── dlg_module.py # Dense Language Guidance (DLG) module
│ └── semidavil.py # Main SemiDAVIL model architecture
├── config.py # Central configuration for all hyperparameters and paths
├── losses.py # Custom loss functions (DyCE, Consistency)
├── train.py # Main training script
├── utils.py # Helper functions (EMA update)
├── requirements.txt # Project dependencies
└── README.md # This file



---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SemiDAVIL.git
cd SemiDAVIL


python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
```



## 📂 File Structure

