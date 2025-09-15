"""
PyTorch Dataset classes for GTA5 and Cityscapes.
Includes label remapping to a common evaluation set.
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# --- Label Mappings ---
# Standard mapping from Cityscapes training IDs to the 19 common evaluation IDs
CITYSCAPES_TRAIN_ID_TO_EVAL_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18
}

# Mapping from GTA5 label IDs to the 19 Cityscapes evaluation IDs
GTA5_ID_TO_EVAL_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18
}

class GTADataset(Dataset):
    """GTA5 Dataset (Source Domain)"""
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))
        self.ignore_index = 255

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        if self.transforms:
            image, label = self.transforms(image, label)

        # Remap GTA5 labels to the common evaluation IDs
        remapped_label = torch.full_like(label, self.ignore_index)
        for gta_id, eval_id in GTA5_ID_TO_EVAL_ID.items():
            remapped_label[label == gta_id] = eval_id

        return image, remapped_label

class CityscapesDataset(Dataset):
    """Cityscapes Dataset (Target Domain)"""
    def __init__(self, root, split='train', mode='labeled', transforms=None, num_labeled=None):
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.ignore_index = 255

        image_base = os.path.join(root, 'leftImg8bit', split)
        label_base = os.path.join(root, 'gtFine', split)
        
        self.image_paths = []
        self.label_paths = []

        for city in sorted(os.listdir(image_base)):
            img_dir = os.path.join(image_base, city)
            lbl_dir = os.path.join(label_base, city)
            for fn in sorted(os.listdir(img_dir)):
                self.image_paths.append(os.path.join(img_dir, fn))
                lbl_fn = fn.replace('leftImg8bit', 'gtFine_labelIds')
                self.label_paths.append(os.path.join(lbl_dir, lbl_fn))

        if split == 'train' and num_labeled is not None:
            # Create a deterministic split for labeled and unlabeled sets
            np.random.RandomState(42).shuffle(self.image_paths)
            np.random.RandomState(42).shuffle(self.label_paths)
            if self.mode == 'labeled':
                self.image_paths = self.image_paths[:num_labeled]
                self.label_paths = self.label_paths[:num_labeled]
            else: # unlabeled
                self.image_paths = self.image_paths[num_labeled:]
                self.label_paths = self.label_paths[num_labeled:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.mode == 'unlabeled':
            # For unlabeled data, we still need a dummy label for transforms
            label = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
            if self.transforms:
                image, _ = self.transforms(image, label)
            return image
        else:
            label_path = self.label_paths[idx]
            label = Image.open(label_path)
            if self.transforms:
                image, label = self.transforms(image, label)
            
            # Remap Cityscapes train IDs to evaluation IDs
            remapped_label = torch.full_like(label, self.ignore_index)
            for train_id, eval_id in CITYSCAPES_TRAIN_ID_TO_EVAL_ID.items():
                remapped_label[label == train_id] = eval_id
            
            return image, remapped_label

# --- Data Augmentation and Transformations ---
class PairedTransforms:
    """Applies the same random augmentations to both image and label."""
    def __init__(self, crop_size=(512, 512)):
        self.crop_size = crop_size
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

    def __call__(self, image, label):
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = T.functional.hflip(image)
            label = T.functional.hflip(label)

        # Apply color jitter to the image only
        image = self.color_jitter(image)

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
        image = T.functional.crop(image, i, j, h, w)
        label = T.functional.crop(label, i, j, h, w)

        # Convert to tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)
        label = torch.from_numpy(np.array(label)).long()

        return image, label

