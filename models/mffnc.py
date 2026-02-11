# models/mffnc.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MFFNC(nn.Module):
    """
    TRUE MFFNC BASELINE
    Flat feature concatenation + MLP
    Outputs both classification (logits) and regression (severity 1–10)
    """

    def __init__(
        self,
        text_dim=384,
        audio_dim=256,
        visual_dim=128,
        stats_dim=5,
        hidden=512
    ):
        super().__init__()

        fusion_dim = text_dim + audio_dim + visual_dim + stats_dim

        # Classification head (depression/no depression)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 2)  # binary classification
        )

        # Regression head (PHQ-9 severity)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1)
        )

    def forward(
        self,
        text_emb=None,
        audio_emb=None,
        visual_emb=None,
        stats_vec=None
    ):
        """
        Inputs:
        text_emb:   [B, 384] - text embeddings
        audio_emb:  [B, 256] - audio features
        visual_emb: [B, 128] - visual features
        stats_vec:  [B, 5] - social media statistics
        
        Returns:
        logits: [B, 2] - classification logits
        severity: [B] - regression output (PHQ-9 severity)
        """
        
        # Infer batch size and device from any non-None input
        B = None
        device = None
        
        if audio_emb is not None:
            B = audio_emb.shape[0]
            device = audio_emb.device
        elif visual_emb is not None:
            B = visual_emb.shape[0]
            device = visual_emb.device
        elif text_emb is not None:
            B = text_emb.shape[0]
            device = text_emb.device
        elif stats_vec is not None:
            B = stats_vec.shape[0]
            device = stats_vec.device
        else:
            raise ValueError("All inputs are None")
        
        # Replace None inputs with zero tensors
        if text_emb is None:
            text_emb = torch.zeros(B, 384, device=device)
        if audio_emb is None:
            audio_emb = torch.zeros(B, 256, device=device)
        if visual_emb is None:
            visual_emb = torch.zeros(B, 128, device=device)
        if stats_vec is None:
            stats_vec = torch.zeros(B, 5, device=device)

        fused = torch.cat(
            [text_emb, audio_emb, visual_emb, stats_vec],
            dim=1
        )

        logits = self.classifier(fused)
        severity = self.regressor(fused).squeeze(1)
        
        return logits, severity
