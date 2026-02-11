import torch
import torch.nn as nn

# -------------------------
# Projection Layer
# -------------------------
class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Cross Attention Block
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, context):
        """
        query:   [B, 1, D]
        context: [B, 1, D]
        """
        attn_out, _ = self.attn(query, context, context)
        return self.norm(query + attn_out)


# -------------------------
# Main Fusion Model
# -------------------------
class DepressionFusionModel(nn.Module):
    """
    Hierarchical Multimodal Fusion Model
    Output: depression severity score (1–10)
    """

    def __init__(self):
        super().__init__()

        # ---- Projections ----
        self.patient_text_proj = Projection(768)
        self.audio_proj = Projection(40)
        self.visual_proj = Projection(52)
        self.relative_text_proj = Projection(768)

        # ---- Patient Fusion ----
        self.text_audio_attn = CrossAttention()
        self.text_av_attn = CrossAttention()

        # ---- Relative Context Injection ----
        self.relative_attn = CrossAttention()

        # ---- Regression Head ----
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        patient_text,
        audio,
        visual,
        relative_text
    ):
        """
        Inputs:
        patient_text:  [B, 768]
        audio:         [B, 88]
        visual:        [B, 52]
        relative_text: [B, 768]
        """

        # ---- Project all modalities ----
        pt = self.patient_text_proj(patient_text).unsqueeze(1)   # [B,1,256]
        a  = self.audio_proj(audio).unsqueeze(1)
        v  = self.visual_proj(visual).unsqueeze(1)
        rt = self.relative_text_proj(relative_text).unsqueeze(1)

        # ---- Patient internal fusion ----
        ta = self.text_audio_attn(pt, a)
        tav = self.text_av_attn(ta, v)

        # ---- Inject relative context ----
        fused = self.relative_attn(tav, rt)

        fused = fused.squeeze(1)  # [B,256]

        # ---- Severity regression ----
        severity = self.regressor(fused)

        return severity.squeeze(1)
