# train.py
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from models.mffnc import MFFNC
from utils.metrics import compute_class_metrics, compute_reg_metrics, expected_calibration_error

TARGET_TEXT_EMB_DIM = 384

class MultimodalDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s

def collate_fn(batch):
    import numpy as np
    texts = [b["text"] for b in batch]
    audio = np.stack([np.array(b["audio_vec"]) for b in batch])
    visual = np.stack([np.array(b["visual_vec"]) for b in batch])
    stats = np.stack([[b["stats"]["neg_prop"], b["stats"]["late_night_ratio"], b["stats"]["posts_per_week"], b["stats"]["std_post_time"], b["stats"]["image_freq"]] for b in batch])
    phq = np.array([b["phq9"] for b in batch], dtype=np.float32)
    label = np.array([b["label"] for b in batch], dtype=np.int64)
    return texts, torch.from_numpy(audio).float(), torch.from_numpy(visual).float(), torch.from_numpy(stats).float(), torch.from_numpy(phq).float(), torch.from_numpy(label)

def train_one_epoch(model, dataloader, optimizer, device, text_encoder=None, text_proj=None, modality_dropout=0.15):
    model.train()
    if text_proj is not None:
        text_proj.train()
    total_loss = 0.0
    for texts, audio, visual, stats, phq, label in tqdm(dataloader, desc="train"):
        batch_size = audio.shape[0]
        # optionally encode text (if using text encoder)
        if text_encoder is not None:
            text_emb = text_encoder.encode(texts)  # numpy
            text_emb = torch.tensor(text_emb).float()
            if text_proj is not None:
                text_emb = text_proj(text_emb.to(device))
        else:
            text_emb = torch.randn(batch_size, TARGET_TEXT_EMB_DIM).to(device)
        text_emb = text_emb.to(device) if text_encoder is not None else text_emb
        audio = audio.to(device)
        visual = visual.to(device)
        stats = stats.to(device)
        phq = phq.to(device)
        label = label.to(device)

        # modality dropout simulation
        if random.random() < modality_dropout:
            # randomly drop one modality
            drop = random.choice(["text", "audio", "visual"])
            if drop == "text": text_emb = None
            if drop == "audio": audio = None
            if drop == "visual": visual = None

        optimizer.zero_grad()
        logits, reg = model(text_emb=text_emb, audio_emb=audio, visual_emb=visual, stats_vec=stats)
        ce = F.cross_entropy(logits, label)
        mae = torch.abs(reg - phq).mean()
        loss = ce + 0.5 * mae
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dl, device, text_encoder=None, text_proj=None):
    model.eval()
    if text_proj is not None:
        text_proj.eval()
    ys, preds, scores, regs, phq_trues = [], [], [], [], []   # collect true phq per-batch
    with torch.no_grad():
        for texts, audio, visual, stats, phh, label in tqdm(dl, desc="eval"):
            B = audio.shape[0]
            if text_encoder is not None:
                text_emb = text_encoder.encode(texts)
                text_emb = torch.tensor(text_emb).float()
                if text_proj is not None:
                    text_emb = text_proj(text_emb.to(device))
                text_emb = text_emb.to(device)
            else:
                text_emb = torch.randn(B, TARGET_TEXT_EMB_DIM).to(device)
            audio, visual, stats = audio.to(device), visual.to(device), stats.to(device)
            logits, reg = model(text_emb=text_emb, audio_emb=audio, visual_emb=visual, stats_vec=stats)
            prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            ys.extend(label.numpy().tolist())
            preds.extend(pred.tolist())
            scores.extend(prob.tolist())
            regs.extend(reg.cpu().numpy().tolist())
            phq_trues.extend(phh.numpy().tolist())   # <- collect ground-truth phq values per-sample
    cmetrics = compute_class_metrics(np.array(ys), np.array(preds), np.array(scores))
    # use phq_trues (same length as regs) instead of loading entire file
    rmetrics = compute_reg_metrics(np.array(phq_trues), np.array(regs))
    ece = expected_calibration_error(np.array(scores), np.array(ys))
    # Also compute validation loss as a scalar for "best model" tracking (optional)
    # We can re-run a quick loss calc here if needed; for simplicity we'll use RMSE as proxy:
    val_loss_proxy = rmetrics.get("rmse", None)
    return cmetrics, rmetrics, ece, val_loss_proxy

def open_dataset_samples(path):
    import json
    res = []
    with open(path, "r") as f:
        for l in f: res.append(json.loads(l))
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_samples.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="save checkpoint every N epochs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ds = MultimodalDataset(args.data)
    # train/test split
    train_idx = list(range(int(0.8*len(ds))))
    test_idx = list(range(int(0.8*len(ds)), len(ds)))
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model once (do NOT instantiate twice)
    model = MFFNC()
    model.to(device)

    # optionally instantiate a text encoder (slow on first run)
    text_enc = None
    try:
        from pipelines.text_extractor import TextEncoder
        text_enc = TextEncoder()
    except Exception as e:
        # If TextEncoder fails to load, we continue with random text embeddings
        print("Warning: TextEncoder couldn't be instantiated:", e)
        text_enc = None

    # probe encoder output dim and create projection if needed
    text_proj = None
    if text_enc is not None:
        sample_text = [ds[0]['text']]
        emb = text_enc.encode(sample_text)
        emb = np.asarray(emb)
        enc_dim = emb.shape[1]
        if enc_dim != TARGET_TEXT_EMB_DIM:
            print(f"Text encoder output dim = {enc_dim}, projecting to {TARGET_TEXT_EMB_DIM}")
            text_proj = nn.Linear(enc_dim, TARGET_TEXT_EMB_DIM).to(device)

    # optimizer must include text_proj params if created
    params = list(model.parameters())
    if text_proj is not None:
        params += list(text_proj.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_ckpt = None

    for e in range(args.epochs):
        loss = train_one_epoch(model, train_dl, opt, device, text_encoder=text_enc, text_proj=text_proj)
        print(f"epoch {e} loss {loss:.4f}")

        # evaluate on validation/test split after each epoch
        cmetrics, rmetrics, ece, val_loss_proxy = evaluate(model, test_dl, device, text_encoder=text_enc, text_proj=text_proj)
        print(f"validation classification: {cmetrics}")
        print(f"validation regression: {rmetrics}")
        print(f"validation ECE: {ece}")

        # Save checkpoint every `save_every` epochs
        if (e + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{e+1}.pt")
            torch.save({
                "epoch": e+1,
                "model_state_dict": model.state_dict(),
                "text_proj_state_dict": text_proj.state_dict() if text_proj is not None else None,
                "optimizer_state_dict": opt.state_dict()
            }, ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")

        # Save best model (by validation loss proxy RMSE if available)
        if val_loss_proxy is not None and val_loss_proxy < best_val_loss:
            best_val_loss = val_loss_proxy
            best_ckpt = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch": e+1,
                "model_state_dict": model.state_dict(),
                "text_proj_state_dict": text_proj.state_dict() if text_proj is not None else None,
                "optimizer_state_dict": opt.state_dict()
            }, best_ckpt)
            print(f"💾 New best model saved to: {best_ckpt} (val proxy loss {best_val_loss:.4f})")

    # final quick eval
    cmetrics, rmetrics, ece, _ = evaluate(model, test_dl, device, text_encoder=text_enc, text_proj=text_proj)
    print("final classification:", cmetrics)
    print("final regression:", rmetrics)
    print("final ECE:", ece)
    if best_ckpt:
        print("Best model checkpoint:", best_ckpt)
    else:
        print("No best checkpoint saved (val proxy missing).")
