# models/semidavil.py
# This file defines the main SemiDAVIL model architecture, integrating all components:
# - Vision and Language Encoders (from CLIP)
# - Captioning Model (BLIP)
# - Dense Language Guidance (DLG) Module
# - Segmentation Decoder

import torch
import torch.nn as nn
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration

from models.dlg_model import DLGModule
import config

class SegmentationHead(nn.Module):
    """A simple segmentation head with a few convolutional layers."""
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class SemiDAVIL(nn.Module):
    """
    The complete SemiDAVIL model as outlined in Figure 2 of the paper.
    It encapsulates the student and teacher networks, along with the frozen
    language encoder and captioning model.
    """
    def __init__(self, num_classes):
        super(SemiDAVIL, self).__init__()
        print("Initializing SemiDAVIL model...")

        # --- Load VLM Encoders (CLIP) ---
        # This corresponds to the "VL Initialization" step in the paper.
        self.clip_model, _ = clip.load(config.VLM_MODEL_NAME, device=config.DEVICE)
        
        # Vision Encoders (Student and Teacher)
        # The paper uses ViT-B/16. We extract the visual part.
        self.student_vision_encoder = self.clip_model.visual
        self.teacher_vision_encoder = self.clip_model.visual
        
        # The language encoder is shared and its weights are frozen.
        self.language_encoder = self.clip_model.transformer
        self.lang_feature_dim = self.language_encoder.width
        
        # --- Load Captioning Model ---
        # The paper uses BLIP-2. We use a smaller BLIP model for accessibility.
        self.caption_processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL_NAME)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL_NAME)
        self.caption_model.to(config.DEVICE)
        
        # --- DLG and Decoder Modules ---
        # The DLG module is a core part of the architecture.
        self.student_dlg = DLGModule(feature_dim=self.lang_feature_dim)
        self.teacher_dlg = DLGModule(feature_dim=self.lang_feature_dim)
        
        # Simple segmentation decoders for student and teacher.
        self.student_decoder = SegmentationHead(in_channels=self.lang_feature_dim, num_classes=num_classes)
        self.teacher_decoder = SegmentationHead(in_channels=self.lang_feature_dim, num_classes=num_classes)

        self._freeze_models()
        self._initialize_teacher()
        print("SemiDAVIL model initialized.")

    def _freeze_models(self):
        """Freeze weights of the language and captioning models as they are not trained."""
        for param in self.language_encoder.parameters():
            param.requires_grad = False
        for param in self.caption_model.parameters():
            param.requires_grad = False
        # Teacher model is not trained via backprop, only EMA
        for param in self.teacher_vision_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_dlg.parameters():
            param.requires_grad = False
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _initialize_teacher(self):
        """Initialize teacher model weights to be identical to the student model's."""
        for teacher_param, student_param in zip(self.teacher_vision_encoder.parameters(), self.student_vision_encoder.parameters()):
            teacher_param.data.copy_(student_param.data)
        for teacher_param, student_param in zip(self.teacher_dlg.parameters(), self.student_dlg.parameters()):
            teacher_param.data.copy_(student_param.data)
        for teacher_param, student_param in zip(self.teacher_decoder.parameters(), self.student_decoder.parameters()):
            teacher_param.data.copy_(student_param.data)

    @torch.no_grad()
    def generate_captions(self, images):
        """Generate text captions for a batch of images."""
        # The paper uses an off-the-shelf captioning model 'C'.
        inputs = self.caption_processor(images=images, return_tensors="pt").to(config.DEVICE)
        generated_ids = self.caption_model.generate(**inputs, max_length=77) # Max length for CLIP
        captions = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
    
    @torch.no_grad()
    def get_lang_features(self, captions):
        """Extract language features using the CLIP text encoder."""
        text_tokens = clip.tokenize(captions).to(config.DEVICE)
        # We need to get the features from the text encoder directly
        lang_features = self.clip_model.encode_text(text_tokens)
        return lang_features.float()

    def forward_student(self, image, lang_features):
        """Forward pass for the student network."""
        # Using ViT, we need to handle its output which is not a feature map.
        # A common practice is to reshape the sequence of patch embeddings.
        # This part requires a careful adaptation of ViT for segmentation.
        # For simplicity, let's assume a simplified feature extraction.
        # A real implementation would use a ViT segmentation adapter.
        vision_features_seq = self.student_vision_encoder(image.type(self.clip_model.dtype))
        
        # Placeholder to convert ViT sequence output to a 2D feature map
        # Real implementation would be more sophisticated (e.g., from HRDA or SegFormer)
        B, N, C = vision_features_seq.shape
        H = W = int((N-1)**0.5) # Assuming square patch grid and excluding CLS token
        vision_features = vision_features_seq[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
        
        multimodal_features = self.student_dlg(vision_features, lang_features.unsqueeze(1))
        logits = self.student_decoder(multimodal_features)
        # Upsample logits to original image size
        logits = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    @torch.no_grad()
    def forward_teacher(self, image, lang_features):
        """Forward pass for the teacher network."""
        self.teacher_vision_encoder.eval()
        self.teacher_dlg.eval()
        self.teacher_decoder.eval()
        
        vision_features_seq = self.teacher_vision_encoder(image.type(self.clip_model.dtype))
        B, N, C = vision_features_seq.shape
        H = W = int((N-1)**0.5)
        vision_features = vision_features_seq[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)

        multimodal_features = self.teacher_dlg(vision_features, lang_features.unsqueeze(1))
        logits = self.teacher_decoder(multimodal_features)
        logits = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def forward(self, labeled_images, unlabeled_images, gt_labels=None):
        """
        Main forward pass to compute losses.
        
        Returns:
            A dictionary containing the supervised loss (dyce_loss) and
            consistency loss (ct_loss).
        """
        # --- Generate Captions and Language Features ---
        # In a real scenario, you might do this once per image, but for simplicity
        # we generate them on the fly for both labeled and unlabeled data.
        all_images = torch.cat([labeled_images, unlabeled_images], dim=0)
        captions = self.generate_captions(all_images)
        lang_features = self.get_lang_features(captions)
        
        labeled_lang_features = lang_features[:labeled_images.size(0)]
        unlabeled_lang_features = lang_features[labeled_images.size(0):]
        
        # --- Supervised Path (Student on Labeled Data) ---
        student_labeled_logits = self.forward_student(labeled_images, labeled_lang_features)
        
        # --- Consistency Path (Student and Teacher on Unlabeled Data) ---
        student_unlabeled_logits = self.forward_student(unlabeled_images, unlabeled_lang_features)
        teacher_unlabeled_logits = self.forward_teacher(unlabeled_images, unlabeled_lang_features)

        return student_labeled_logits, student_unlabeled_logits, teacher_unlabeled_logits
