import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    use_baseline=False
):
    model.eval()
    total_loss = 0.0

    preds_all = []
    targets_all = []

    for batch in tqdm(dataloader, desc="Val", leave=False):
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
        total_loss += loss.item()

        preds_all.append(preds.cpu().numpy())
        targets_all.append(batch["severity"].cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    mae = np.mean(np.abs(preds_all - targets_all))
    rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))

    return total_loss / len(dataloader), mae, rmse
