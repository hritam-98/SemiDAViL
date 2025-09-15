"""
Main training script for the SemiDAVIL framework.
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import os

# It's good practice to import config as a module to avoid namespace pollution
import config
from models.semidavil import SemiDAVIL
from losses import DyCELoss, ConsistencyLoss
from utils import update_teacher_model
from data.loader import get_data_loaders

def train():
    """Main training loop for SemiDAVIL."""
    print("--- Initializing SemiDAVIL Training ---")
    print(f"Using device: {config.DEVICE}")

    # --- Setup ---
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    # --- Data Loaders ---
    source_loader, target_labeled_loader, target_unlabeled_loader = get_data_loaders(config)
    
    # Use iterators for continuous sampling
    source_iter = iter(source_loader)
    target_labeled_iter = iter(target_labeled_loader)
    target_unlabeled_iter = iter(target_unlabeled_loader)

    # --- Model, Optimizer, Losses ---
    model = SemiDAVIL(num_classes=config.NUM_CLASSES, device=config.DEVICE).to(config.DEVICE)
    optimizer = optim.AdamW(
        model.student_network.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    dyce_loss_fn = DyCELoss(
        num_classes=config.NUM_CLASSES,
        hard_percentage=config.DYCE_HARD_PERCENTAGE,
        omega=config.DYCE_OMEGA
    )
    consistency_loss_fn = ConsistencyLoss(threshold=config.PSEUDO_LABEL_THRESHOLD)

    print("\n--- Starting Training Loop ---")
    for i in tqdm(range(config.NUM_ITERATIONS), desc="Training Iterations"):
        model.train()
        optimizer.zero_grad()
        
        # --- Fetch Data Batches ---
        try:
            source_images, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_images, source_labels = next(source_iter)
        
        try:
            target_labeled_images, target_labels = next(target_labeled_iter)
        except StopIteration:
            target_labeled_iter = iter(target_labeled_loader)
            target_labeled_images, target_labels = next(target_labeled_iter)
        
        try:
            target_unlabeled_images = next(target_unlabeled_iter)
        except StopIteration:
            target_unlabeled_iter = iter(target_unlabeled_loader)
            target_unlabeled_images = next(target_unlabeled_iter)

        # Move data to the configured device
        source_images, source_labels = source_images.to(config.DEVICE), source_labels.to(config.DEVICE)
        target_labeled_images, target_labels = target_labeled_images.to(config.DEVICE), target_labels.to(config.DEVICE)
        target_unlabeled_images = target_unlabeled_images.to(config.DEVICE)
        
        # --- Supervised Loss Calculation ---
        labeled_images = torch.cat([source_images, target_labeled_images])
        labeled_targets = torch.cat([source_labels, target_labels])
        student_labeled_logits = model(labeled_images, network='student')
        supervised_loss = dyce_loss_fn(student_labeled_logits, labeled_targets)

        # --- Consistency Loss Calculation ---
        teacher_unlabeled_logits = model(target_unlabeled_images, network='teacher')
        student_unlabeled_logits = model(target_unlabeled_images, network='student')
        consistency_loss = consistency_loss_fn(student_unlabeled_logits, teacher_unlabeled_logits)
        
        # --- Total Loss and Optimization ---
        total_loss = supervised_loss + consistency_loss
        total_loss.backward()
        optimizer.step()

        # --- Update Teacher Model using EMA ---
        update_teacher_model(model.student_network, model.teacher_network, alpha=config.EMA_ALPHA)

        if (i + 1) % 100 == 0:
            tqdm.write(f"Iter [{i+1}/{config.NUM_ITERATIONS}] | "
                       f"Total Loss: {total_loss.item():.4f} | "
                       f"Supervised (DyCE): {supervised_loss.item():.4f} | "
                       f"Consistency: {consistency_loss.item():.4f}")

    # --- Save Final Model ---
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "semidavil_final.pth")
    torch.save(model.student_network.state_dict(), final_model_path)
    print(f"\n--- Training Finished ---")
    print(f"Final student model saved to {final_model_path}")

if __name__ == "__main__":
    train()

