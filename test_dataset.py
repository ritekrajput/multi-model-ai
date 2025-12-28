from dataset.loader import get_dataloader

loader = get_dataloader(
    csv_path="data/metadata.csv",
    audio_dir="data/audio",
    openface_csv_dir="data/openface_csv",
    batch_size=2
)

batch = next(iter(loader))

print("Patient text:", batch["patient_text"].shape)   # [B, 768]
print("Relative text:", batch["relative_text"].shape) # [B, 768]
print("Audio:", batch["audio"].shape)
print("Visual:", batch["visual"].shape)               # [B, 52]
print("Severity:", batch["severity"])
