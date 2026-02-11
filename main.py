import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data.loader import get_dataloader
from models.fusion_model import DepressionFusionModel
from models.mffnc import MFFNC


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 4

USE_BASELINE = False   # True → MFFNC | False → Main Fusion Model

DATA_CSV = "data/metadata.csv"
AUDIO_DIR = "data/audio"
OPENFACE_DIR = "data/openface_csv"

MODEL_SAVE_PATH = "best_model.pt"


# -----------------------------
# TRAIN ONE EPOCH
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        if USE_BASELINE:
            preds = model(
                batch["patient_text"],
                batch["audio"],
                batch["visual"],
                batch["relative_text"]
            )
        else:
            preds = model(
                patient_text=batch["patient_text"],
                audio=batch["audio"],
                visual=batch["visual"],
                relative_text=batch["relative_text"]
            )

        loss = criterion(preds, batch["severity"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# VALIDATION
# -----------------------------
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Validation"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        if USE_BASELINE:
            preds = model(
                batch["patient_text"],
                batch["audio"],
                batch["visual"],
                batch["relative_text"]
            )
        else:
            preds = model(
                patient_text=batch["patient_text"],
                audio=batch["audio"],
                visual=batch["visual"],
                relative_text=batch["relative_text"]
            )

        loss = criterion(preds, batch["severity"])
        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Using device: {DEVICE}")
    print("Model:", "MFFNC Baseline" if USE_BASELINE else "Main Fusion Model")

    # Data loaders
    train_loader = get_dataloader(
        csv_path=DATA_CSV,
        audio_dir=AUDIO_DIR,
        openface_csv_dir=OPENFACE_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        device=DEVICE
    )

    val_loader = get_dataloader(
        csv_path=DATA_CSV,
        audio_dir=AUDIO_DIR,
        openface_csv_dir=OPENFACE_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=DEVICE
    )

    # Model
    if USE_BASELINE:
        model = MFFNC().to(DEVICE)
    else:
        model = DepressionFusionModel().to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best model saved")

    print("Training complete.")


if __name__ == "__main__":
    main()
