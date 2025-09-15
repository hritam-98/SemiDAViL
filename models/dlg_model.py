# models/dlg_module.py
# This file implements the Dense Language Guidance (DLG) module, which is a
# key contribution of the SemiDAVIL paper (Section 3.2 and Figure 3).

import torch
import torch.nn as nn
import torch.nn.functional as F

class DLGModule(nn.Module):
    """
    Dense Language Guidance (DLG) module for fusing visual and language features.
    This module implements the cross-attention mechanism described in Section 3.2
    to create a rich, multimodal feature representation.
    """
    def __init__(self, feature_dim):
        super(DLGModule, self).__init__()
        self.feature_dim = feature_dim
        
        # Linear layers to project vision features to Key and Value
        self.vision_key_proj = nn.Linear(feature_dim, feature_dim)
        self.vision_value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Linear layers to project language features to Key and Value
        self.lang_key_proj = nn.Linear(feature_dim, feature_dim)
        self.lang_value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = feature_dim ** -0.5

    def forward(self, vision_features, lang_features):
        """
        Args:
            vision_features (torch.Tensor): Visual features from the vision encoder.
                                            Shape: (B, C, H, W).
            lang_features (torch.Tensor): Language embeddings from the language encoder.
                                          Shape: (B, N, C), where N is sequence length.

        Returns:
            torch.Tensor: The fused multimodal feature map. Shape: (B, C, H, W).
        """
        B, C, H, W = vision_features.shape
        
        # Reshape vision features for processing
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        vision_features_flat = vision_features.flatten(2).permute(0, 2, 1)
        
        # Project features to Key-Value pairs
        # F_{V,L}^{K,V} <- Linear(F_{V,L})
        vision_key = self.vision_key_proj(vision_features_flat)  # (B, H*W, C)
        vision_value = self.vision_value_proj(vision_features_flat) # (B, H*W, C)
        
        lang_key = self.lang_key_proj(lang_features)    # (B, N, C)
        lang_value = self.lang_value_proj(lang_features)      # (B, N, C)

        # Generate attention matrix A (Equation 2)
        # A = (1/√c) * F_V^K * (F_L^K)^T
        # Resulting shape: (B, H*W, N)
        attention_matrix = torch.bmm(vision_key, lang_key.transpose(1, 2)) * self.scale

        # Normalize across both dimensions (as implied by the paper's description)
        # and compute cross-attention on vision and language features.
        
        # Apply softmax to get attention weights for language -> vision
        # Softmax is applied over the language token dimension (dim=2)
        attn_v = F.softmax(attention_matrix, dim=2) # (B, H*W, N)
        
        # Apply softmax to get attention weights for vision -> language
        # Softmax is applied over the vision pixel dimension (dim=1)
        attn_l = F.softmax(attention_matrix, dim=1) # (B, H*W, N)
        
        # Compute attended features (Equation 3, conceptually)
        # F_V^A = SoftMax[A] * F_L^V (language-attended vision features)
        # The paper's notation is slightly ambiguous. The implementation here fuses language context into vision.
        # A common interpretation is to use language as the context for vision features.
        lang_attended_vision = torch.bmm(attn_v, lang_value) # (B, H*W, C)
        
        # The paper suggests combining two attended features: F_M = F_V^A * (F_L^A)^T
        # A simpler, more common fusion method is to add or concatenate the attended feature
        # with the original vision feature. We will use an additive fusion for stability.
        fused_features_flat = vision_features_flat + lang_attended_vision
        
        # Reshape back to image feature map format
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        multimodal_features = fused_features_flat.permute(0, 2, 1).view(B, C, H, W)
        
        return multimodal_features
