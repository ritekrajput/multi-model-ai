from torch.utils.data import DataLoader
from data.multimodel_dataset import DepressionDataset
from data.collate import depression_collate

def get_dataloader(
    csv_path,
    audio_dir,
    openface_csv_dir,
    batch_size=4,
    shuffle=True,
    device="cpu"
):
    dataset = DepressionDataset(
        csv_path=csv_path,
        audio_dir=audio_dir,
        openface_csv_dir=openface_csv_dir,
        device=device
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=depression_collate,
        num_workers=0
    )
