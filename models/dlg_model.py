"""
Implementation of the Dense Language Guidance (DLG) module from Figure 3.
This module performs deep cross-modal fusion of visual and language features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLanguageGuidance(nn.Module):
    """
    Fuses visual and language features using a symmetric cross-attention mechanism.
    This corresponds to the DLG block in the paper's architecture diagram.
    """
    def __init__(self, feature_dim):
        super(DenseLanguageGuidance, self).__init__()
        self.feature_dim = feature_dim

        # Linear projections for query, key, value for both modalities
        self.vis_query_proj = nn.Linear(feature_dim, feature_dim)
        self.vis_key_proj = nn.Linear(feature_dim, feature_dim)
        self.vis_value_proj = nn.Linear(feature_dim, feature_dim)

        self.lang_query_proj = nn.Linear(feature_dim, feature_dim)
        self.lang_key_proj = nn.Linear(feature_dim, feature_dim)
        self.lang_value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = feature_dim ** -0.5

    def forward(self, vis_features, lang_features):
        """
        Args:
            vis_features (torch.Tensor): Visual features of shape (B, H*W, C).
            lang_features (torch.Tensor): Language features of shape (B, N, C),
                                          where N is the number of tokens.
        
        Returns:
            torch.Tensor: Fused multimodal features of shape (B, H*W, C).
        """
        B, HW, C = vis_features.shape
        _, N, _ = lang_features.shape

        # Project features into key and value pairs
        vis_key = self.vis_key_proj(vis_features)   # (B, HW, C)
        lang_key = self.lang_key_proj(lang_features) # (B, N, C)
        
        vis_value = self.vis_value_proj(vis_features) # (B, HW, C)
        lang_value = self.lang_value_proj(lang_features) # (B, N, C)

        # --- Symmetric Cross-Attention (Equations 2 & 3) ---

        # 1. Generate attention matrix A
        # (B, HW, C) @ (B, C, N) -> (B, HW, N)
        attn_matrix = (vis_key @ lang_key.transpose(-2, -1)) * self.scale
        attn_matrix = F.softmax(attn_matrix, dim=-1)

        # 2. Compute language-attended vision features (F_V^A)
        # (B, HW, N) @ (B, N, C) -> (B, HW, C)
        lang_attended_vis = attn_matrix @ lang_value
        
        # 3. Compute vision-attended language features (F_L^A)
        # We need to compute attention in the other direction as well.
        # Transpose attn_matrix to get (B, N, HW)
        vision_attended_lang = attn_matrix.transpose(-2, -1) @ vis_value
        
        # For segmentation, we need to combine these back to the visual feature space.
        # We project the vision-attended language features back to the visual space.
        # (B, N, HW)^T @ (B, N, C) -> (B, HW, C)
        projected_vision_attended_lang = attn_matrix @ vision_attended_lang
        
        # Fuse the features. A simple addition or concatenation followed by a linear layer works well.
        fused_features = lang_attended_vis + projected_vision_attended_lang
        
        return fused_features

