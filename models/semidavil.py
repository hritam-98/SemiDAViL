"""
Main SemiDAVIL model architecture, combining the student-teacher networks,
CLIP encoders, DLG module, and a segmentation decoder.
"""
import torch
import torch.nn as nn
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from .dlg_module import DenseLanguageGuidance

class SegmentationDecoder(nn.Module):
    """A simple decoder for semantic segmentation."""
    def __init__(self, in_channels, num_classes):
        super(SegmentationDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x) # Upsample to original image size
        return x

class SemiDAVIL(nn.Module):
    """
    The complete SemiDAVIL model with student and teacher networks.
    """
    def __init__(self, num_classes, vlm_model_name="ViT-B/16", caption_model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        super(SemiDAVIL, self).__init__()
        self.device = device

        # --- Vision and Language Encoders (from CLIP) ---
        print(f"Loading CLIP model: {vlm_model_name}...")
        self.clip_model, self.clip_preprocess = clip.load(vlm_model_name, device=self.device)
        self.vis_encoder = self.clip_model.visual
        self.lang_encoder = self.clip_model.transformer
        self.feature_dim = self.vis_encoder.output_dim
        
        # --- Captioning Model (from Hugging Face) ---
        print(f"Loading Captioning model: {caption_model_name}...")
        self.caption_processor = Blip2Processor.from_pretrained(caption_model_name)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(caption_model_name).to(self.device)

        # --- Student Network ---
        self.student_dlg = DenseLanguageGuidance(self.feature_dim)
        self.student_decoder = SegmentationDecoder(self.feature_dim, num_classes)
        self.student_network = nn.ModuleDict({
            'dlg': self.student_dlg,
            'decoder': self.student_decoder
        })

        # --- Teacher Network (identical architecture) ---
        self.teacher_dlg = DenseLanguageGuidance(self.feature_dim)
        self.teacher_decoder = SegmentationDecoder(self.feature_dim, num_classes)
        self.teacher_network = nn.ModuleDict({
            'dlg': self.teacher_dlg,
            'decoder': self.teacher_decoder
        })

        # Initialize teacher with student weights and stop its gradients
        self._initialize_teacher()

    def _initialize_teacher(self):
        for student_params, teacher_params in zip(self.student_network.parameters(), self.teacher_network.parameters()):
            teacher_params.data.copy_(student_params.data)
            teacher_params.requires_grad = False # Teacher is not trained via backprop

    def forward(self, images, network='student'):
        B, _, H, W = images.shape
        
        # --- 1. Get Visual Features ---
        # CLIP ViT uses a patch-based approach
        vis_features = self.vis_encoder(images) # (B, HW+1, C)
        vis_features = vis_features[:, 1:, :] # Exclude CLS token -> (B, HW, C)
        
        # --- 2. Generate Captions and Language Features ---
        with torch.no_grad():
            captions = self._generate_captions(images)
            text_tokens = clip.tokenize(captions).to(self.device)
            lang_features = self.clip_model.encode_text(text_tokens) # (B, C)
            # Unsqueeze to add token dimension for DLG module
            lang_features = lang_features.unsqueeze(1) # (B, 1, C)

        # --- 3. Select Network and Fuse Features ---
        if network == 'student':
            fused_features = self.student_dlg(vis_features, lang_features)
            decoder = self.student_decoder
        elif network == 'teacher':
            with torch.no_grad():
                fused_features = self.teacher_dlg(vis_features, lang_features)
            decoder = self.teacher_decoder
        else:
            raise ValueError("network must be 'student' or 'teacher'")
            
        # --- 4. Decode for Segmentation Map ---
        # Reshape fused features back to a 2D grid for the decoder
        # ViT patch size is 16 for ViT-B/16
        patch_size = self.vis_encoder.conv1.kernel_size[0]
        h, w = H // patch_size, W // patch_size
        fused_features = fused_features.permute(0, 2, 1).reshape(B, self.feature_dim, h, w)
        
        logits = decoder(fused_features)
        
        return logits

    def _generate_captions(self, images):
        """Generates a caption for each image in the batch."""
        inputs = self.caption_processor(images=images, return_tensors="pt").to(self.device)
        generated_ids = self.caption_model.generate(**inputs)
        captions = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [c.strip() for c in captions]

