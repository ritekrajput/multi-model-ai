import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipelines.text_extractor import TextEncoder
from pipelines.visual_extractor import parse_openface_csv
from pipelines.audio_extractor import extract_audio_features

class DepressionDataset(Dataset):
    def __init__(self, csv_path, audio_dir, openface_csv_dir, device="cpu"):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.openface_csv_dir = openface_csv_dir

        self.patient_text_encoder = TextEncoder(device=device)
        self.relative_text_encoder = TextEncoder(device=device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        # Patient text
        patient_text = torch.tensor(
            self.patient_text_encoder.encode([row["patient_text"]])[0],
            dtype=torch.float32
        )

        # Relative text
        relative_text = torch.tensor(
            self.relative_text_encoder.encode([row["relative_text"]])[0],
            dtype=torch.float32
        )

        # Audio (raw → features on-the-fly)
        audio_path = os.path.join(self.audio_dir, f"{sample_id}.wav")
        audio_feat = extract_audio_features(audio_path)
        audio_feat = torch.tensor(audio_feat, dtype=torch.float32)

        # Visual
        csv_path = os.path.join(self.openface_csv_dir, f"{sample_id}.csv")
        if os.path.exists(csv_path):
            visual = parse_openface_csv(csv_path)
        else:
            visual = torch.zeros(52)

        visual = torch.tensor(visual, dtype=torch.float32)

        severity = torch.tensor(row["severity"], dtype=torch.float32)

        return {
            "patient_text": patient_text,
            "relative_text": relative_text,
            "audio": audio_feat,
            "visual": visual,
            "severity": severity
        }
