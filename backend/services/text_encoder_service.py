import torch
from pipelines.text_extractor import TextEncoder

class TextEncoderService:
    """
    Loads the text encoder once and reuses it for all API requests
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.encoder = TextEncoder(device=device)

    def encode_patient_and_relative(self, patient_text: str, relative_text: str):
        """
        Returns:
        - patient_emb: torch.Tensor [1, 768]
        - relative_emb: torch.Tensor [1, 768]
        """
        embeddings = self.encoder.encode(
            [patient_text, relative_text]
        )

        patient_emb = torch.tensor(embeddings[0]).unsqueeze(0)
        relative_emb = torch.tensor(embeddings[1]).unsqueeze(0)

        return patient_emb, relative_emb
