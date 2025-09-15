# utils.py
# This file contains helper functions used across the project, such as the
# Exponential Moving Average (EMA) update mechanism for the teacher model.

import torch
import logging

def setup_logger():
    """Sets up a basic logger."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

@torch.no_grad()
def update_teacher_model_ema(student_model, teacher_model, alpha):
    """
    Updates the teacher model's parameters using an Exponential Moving Average (EMA)
    of the student model's parameters. This is a core component of the student-teacher
    framework for consistency training.

    This function implements Equation 6 from the paper:
    θ_T(t) ← α * θ_T(t-1) + (1 - α) * θ_S(t)

    Args:
        student_model (torch.nn.Module): The student model, which is being trained with backpropagation.
        teacher_model (torch.nn.Module): The teacher model, which is the target for the EMA update.
        alpha (float): The momentum coefficient for the EMA update.
    """
    # Iterate over the parameters of both models
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        # Apply the EMA update rule
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

    # Also update buffer states (e.g., running means and variances in batch norm layers)
    for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)
