# train.py
# This is the main script to run the training process for the SemiDAVIL model.

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # Using dummy data for demonstration
import torch.nn.functional as F
import copy

import config
from models.semidavil import SemiDAVIL
from losses import DyCELoss, ConsistencyLoss
from utils import update_teacher_model_ema, setup_logger

def main():
    logger = setup_logger()
    logger.info("Starting SemiDAVIL training process...")
    logger.info(f"Using device: {config.DEVICE}")

    # --- 1. Model Initialization ---
    model = SemiDAVIL(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Separate parameters for student and teacher for clarity in optimizer
    student_params = [
        {'params': model.student_vision_encoder.parameters()},
        {'params': model.student_dlg.parameters()},
        {'params': model.student_decoder.parameters()}
    ]

    # --- 2. Optimizer and Schedulers ---
    optimizer = optim.AdamW(
        student_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # The paper mentions exponential decay
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LR_DECAY_FACTOR)

    # --- 3. Loss Functions ---
    dyce_loss_fn = DyCELoss(
        num_classes=config.NUM_CLASSES,
        hard_percentage=config.DYCE_HARD_PERCENTAGE,
        omega=config.DYCE_OMEGA
    )
    consistency_loss_fn = ConsistencyLoss(threshold=config.PSEUDO_LABEL_THRESHOLD)

    # --- 4. Dataloaders (Using Dummy Data) ---
    # In a real implementation, replace this with the dataset class from dataset.py
    # that loads GTA5/Cityscapes data.
    logger.info("Setting up dummy dataloaders for demonstration.")
    # Assuming CROP_SIZE for all inputs for simplicity
    H, W = config.CROP_SIZE
    # Create a mix of labeled (source + target) and unlabeled target data
    # Let's say batch size is 4, with 2 labeled and 2 unlabeled
    labeled_data = torch.randn(config.BATCH_SIZE // 2, 3, H, W)
    labeled_targets = torch.randint(0, config.NUM_CLASSES, (config.BATCH_SIZE // 2, H, W))
    unlabeled_data = torch.randn(config.BATCH_SIZE // 2, 3, H, W)
    
    dummy_dataset = TensorDataset(labeled_data, labeled_targets, unlabeled_data)
    # Use a persistent dataloader to simulate iterating for N steps
    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKERS, persistent_workers=True)
    data_iterator = iter(dataloader)
    
    # --- 5. Training Loop ---
    model.train()
    for i in range(config.NUM_ITERATIONS):
        try:
            labeled_images, gt_labels, unlabeled_images = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader) # Reset iterator
            labeled_images, gt_labels, unlabeled_images = next(data_iterator)

        # Move data to device
        labeled_images = labeled_images.squeeze(0).to(config.DEVICE)
        gt_labels = gt_labels.squeeze(0).to(config.DEVICE)
        unlabeled_images = unlabeled_images.squeeze(0).to(config.DEVICE)

        # --- Forward Pass ---
        student_labeled_logits, student_unlabeled_logits, teacher_unlabeled_logits = model(
            labeled_images, unlabeled_images
        )
        
        # --- Loss Calculation ---
        # Supervised loss on labeled data
        loss_dyce = dyce_loss_fn(student_labeled_logits, gt_labels)
        
        # Consistency loss on unlabeled data
        loss_ct = consistency_loss_fn(student_unlabeled_logits, teacher_unlabeled_logits.detach())
        
        total_loss = loss_dyce + loss_ct

        # --- Backward Pass & Optimization ---
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Update Teacher Model using EMA ---
        # A deepcopy is needed for the first update, then we pass the model itself
        # This implementation detail can be abstracted into the model class.
        # For clarity, here we get the student and teacher components to update.
        student_components = nn.ModuleList([
            model.student_vision_encoder,
            model.student_dlg,
            model.student_decoder
        ])
        teacher_components = nn.ModuleList([
            model.teacher_vision_encoder,
            model.teacher_dlg,
            model.teacher_decoder
        ])
        update_teacher_model_ema(student_components, teacher_components, config.EMA_ALPHA)

        if (i + 1) % config.LOG_INTERVAL == 0:
            logger.info(
                f"Iteration [{i+1}/{config.NUM_ITERATIONS}], "
                f"Total Loss: {total_loss.item():.4f}, "
                f"DyCE Loss: {loss_dyce.item():.4f}, "
                f"Consistency Loss: {loss_ct.item():.4f}"
            )
            
    logger.info("Training finished.")
    # Save model checkpoint
    torch.save(model.state_dict(), f"{config.CHECKPOINT_DIR}/semidavil_final.pth")
    logger.info(f"Model saved to {config.CHECKPOINT_DIR}/semidavil_final.pth")

if __name__ == "__main__":
    # Create checkpoint directory if it doesn't exist
    import os
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
    main()
