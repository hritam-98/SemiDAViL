# losses.py
# This file implements the custom loss functions described in the SemiDAVIL paper:
# 1. DyCELoss: Dynamic Cross-Entropy loss for handling class imbalance (Section 3.4).
# 2. ConsistencyLoss: Loss for enforcing consistency between student and teacher predictions
#    on unlabeled data, using a confidence threshold (Section 3.3).

import torch
import torch.nn as nn
import torch.nn.functional as F

class DyCELoss(nn.Module):
    """
    Implementation of the Dynamic Cross-Entropy (DyCE) Loss from Section 3.4 of the paper.
    This loss function dynamically re-weights gradients based on the class distribution
    within a mined subset of the hardest examples in each mini-batch.

    This implements Equation 9:
    L_DyCE = - (1/f_H^ω) * Σ_c [ (1/f_c^(1-ω)) * Σ_{i in H} [y_{i,c} * log(p_{i,c})] ]
    """
    def __init__(self, num_classes, hard_percentage=0.2, omega=0.5, ignore_index=-1):
        super(DyCELoss, self).__init__()
        self.num_classes = num_classes
        self.hard_percentage = hard_percentage
        self.omega = omega
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): The model's predictions (B, C, H, W).
            targets (torch.Tensor): The ground truth labels (B, H, W).

        Returns:
            torch.Tensor: The calculated DyCE loss.
        """
        # 1. Compute standard CE loss for each sample/pixel
        per_pixel_loss = self.ce_loss(logits, targets)
        
        # Flatten the loss and targets to work with them on a per-pixel basis
        flat_loss = per_pixel_loss.view(-1)
        flat_targets = targets.view(-1)
        
        # Mask out ignored pixels
        valid_mask = flat_targets != self.ignore_index
        valid_loss = flat_loss[valid_mask]
        valid_targets = flat_targets[valid_mask]

        if valid_loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # 2. Create a subset H from the batch (hardest h% instances)
        num_hard = int(self.hard_percentage * valid_loss.numel())
        if num_hard == 0:
            return torch.tensor(0.0, device=logits.device)
            
        hard_losses, hard_indices = torch.topk(valid_loss, k=num_hard)
        hard_targets = valid_targets[hard_indices]

        # 3. Calculate class frequencies (f_c) and subset size (f_H)
        f_H = float(num_hard)
        class_counts = torch.bincount(hard_targets, minlength=self.num_classes).float()
        
        # To avoid division by zero for classes not in the hard set
        f_c = class_counts + 1e-8

        # 4. Calculate dynamic class weights and volume weight
        class_weights = 1.0 / (f_c ** (1.0 - self.omega))
        volume_weight = 1.0 / (f_H ** self.omega)
        
        # Reweight the hard losses
        weighted_losses = class_weights[hard_targets] * hard_losses

        # 5. Compute the final DyCE loss
        dyce_loss = volume_weight * torch.sum(weighted_losses)

        return dyce_loss


class ConsistencyLoss(nn.Module):
    """
    Implementation of the Consistency Training (CT) loss from Section 3.3 of the paper.
    It enforces consistency between student predictions and high-confidence teacher predictions
    on unlabeled data.

    This implements Equation 4:
    L_CT = Σ_{p where max(y_p^T) >= Th} CE(y_p^S, y_p^T_label)
    """
    def __init__(self, threshold=0.95, ignore_index=-1):
        super(ConsistencyLoss, self).__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        # Using KL Divergence is a common and stable way to implement this loss
        # It's equivalent to CE when the target is one-hot
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits (torch.Tensor): Predictions from the student model (B, C, H, W).
            teacher_logits (torch.Tensor): Predictions from the teacher model (B, C, H, W).

        Returns:
            torch.Tensor: The calculated consistency loss.
        """
        student_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Get pseudo-labels and confidence scores from the teacher
        confidence, pseudo_labels = torch.max(teacher_probs, dim=1)
        
        # Create a mask for high-confidence predictions
        mask = (confidence >= self.threshold).float()
        
        # Calculate loss only for masked pixels
        loss_matrix = self.criterion(student_probs, teacher_probs)
        
        # We need to sum over the class dimension
        loss_per_pixel = torch.sum(loss_matrix, dim=1)
        
        # Apply the mask
        masked_loss = loss_per_pixel * mask
        
        # Average the loss over the number of confident pixels
        if mask.sum() > 0:
            return masked_loss.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=student_logits.device)
