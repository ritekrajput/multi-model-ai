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
    Regression only (severity 1–10)
    """

    def __init__(
        self,
        text_dim=768,
        audio_dim=88,
        visual_dim=52,
        relative_dim=768,
        hidden=512
    ):
        super().__init__()

        fusion_dim = text_dim + audio_dim + visual_dim + relative_dim

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

        fused = torch.cat(
            [patient_text, audio, visual, relative_text],
            dim=1
        )

        severity = self.regressor(fused)
        return severity.squeeze(1)
