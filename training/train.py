import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_baseline=False
):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        if use_baseline:
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

        running_loss += loss.item()

    return running_loss / len(dataloader)
