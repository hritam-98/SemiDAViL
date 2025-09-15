"""
Factory function to create data loaders for the SSDA task.
"""
import torch
from torch.utils.data import DataLoader
from .datasets import GTADataset, CityscapesDataset, PairedTransforms

def get_data_loaders(config):
    """
    Creates and returns the data loaders for source and target domains.

    Args:
        config: A module containing configuration parameters.

    Returns:
        A tuple of DataLoaders: (source_labeled_loader, target_labeled_loader, target_unlabeled_loader)
    """
    transforms = PairedTransforms(crop_size=(512, 512))

    # --- Source DataLoader (GTA5) ---
    source_dataset = GTADataset(root=config.GTA5_DATA_PATH, transforms=transforms)
    source_labeled_loader = DataLoader(
        source_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # --- Target DataLoaders (Cityscapes) ---
    # Labeled set
    target_labeled_dataset = CityscapesDataset(
        root=config.CITYSCAPES_DATA_PATH,
        split='train',
        mode='labeled',
        transforms=transforms,
        num_labeled=config.NUM_LABELED_TARGET_SAMPLES
    )
    target_labeled_loader = DataLoader(
        target_labeled_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # Unlabeled set
    target_unlabeled_dataset = CityscapesDataset(
        root=config.CITYSCAPES_DATA_PATH,
        split='train',
        mode='unlabeled',
        transforms=transforms,
        num_labeled=config.NUM_LABELED_TARGET_SAMPLES
    )
    target_unlabeled_loader = DataLoader(
        target_unlabeled_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    print(f"Source Labeled (GTA5) loader created with {len(source_dataset)} images.")
    print(f"Target Labeled (Cityscapes) loader created with {len(target_labeled_dataset)} images.")
    print(f"Target Unlabeled (Cityscapes) loader created with {len(target_unlabeled_dataset)} images.")

    return source_labeled_loader, target_labeled_loader, target_unlabeled_loader

