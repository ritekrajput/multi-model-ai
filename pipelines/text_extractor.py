from transformers import AutoTokenizer, AutoModel
import torch

class TextEncoder:
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, max_length=128):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # CLS token
        emb = outputs.last_hidden_state[:, 0, :]
        return emb.cpu().numpy()
