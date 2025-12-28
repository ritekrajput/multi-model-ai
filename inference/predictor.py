import torch
from models.fusion_model import DepressionFusionModel

class DepressionPredictor:
    """
    Inference-only wrapper around trained fusion model
    """

    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = DepressionFusionModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        patient_text,
        relative_text,
        audio=None,
        visual=None
    ):
        """
        All inputs are torch tensors
        Audio / visual can be None (zero-filled)
        """

        B = patient_text.shape[0]

        if audio is None:
            audio = torch.zeros((B, 88), device=self.device)

        if visual is None:
            visual = torch.zeros((B, 52), device=self.device)

        severity = self.model(
            patient_text=patient_text,
            audio=audio,
            visual=visual,
            relative_text=relative_text
        )

        # Clamp ONLY at inference
        severity = torch.clamp(severity, 1.0, 10.0)

        return severity
