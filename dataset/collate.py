import torch

def depression_collate(batch):
    return {
        "patient_text": torch.stack([b["patient_text"] for b in batch]),
        "relative_text": torch.stack([b["relative_text"] for b in batch]),
        "audio": torch.stack([b["audio"] for b in batch]),
        "visual": torch.stack([b["visual"] for b in batch]),
        "severity": torch.stack([b["severity"] for b in batch])
    }
