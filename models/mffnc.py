# models/mffnc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, q, kv, q_mask=None, kv_mask=None):
        # q: (B, Lq, D), kv: (B, Lkv, D)
        attn_out, _ = self.mha(query=q, key=kv, value=kv, key_padding_mask=None)
        q = self.ln1(q + attn_out)
        ff = self.ff(q)
        q = self.ln2(q + ff)
        return q

class FusionCrossAttention(nn.Module):
    def __init__(self, dim_text=256, dim_audio=256, dim_visual=256, fusion_dim=256):
        super().__init__()
        # projectors (input dims -> fusion_dim)
        self.pt = nn.Linear(dim_text, fusion_dim)
        self.pa = nn.Linear(dim_audio, fusion_dim)
        self.pv = nn.Linear(dim_visual, fusion_dim)
        # cross-attention blocks: text<-audio/visual, audio<-visual/text, visual<-audio/text (pairwise)
        self.ta = CrossAttentionBlock(fusion_dim)
        self.tv = CrossAttentionBlock(fusion_dim)
        self.at = CrossAttentionBlock(fusion_dim)
        self.av = CrossAttentionBlock(fusion_dim)
        self.vt = CrossAttentionBlock(fusion_dim)
        self.va = CrossAttentionBlock(fusion_dim)
        # gating (text/audio/visual)
        self.gate = nn.Parameter(torch.ones(3))  # text/audio/visual gates
        self.fusion_dim = fusion_dim

    def forward(self, text_vec, audio_vec, visual_vec, mask=(True,True,True)):
        # All inputs are (B, D) or None. Convert to (B,1,D) sequences or zeros when missing.
        device = next(self.parameters()).device
        B = None
        if text_vec is not None:
            B = text_vec.shape[0]
        elif audio_vec is not None:
            B = audio_vec.shape[0]
        elif visual_vec is not None:
            B = visual_vec.shape[0]
        else:
            raise ValueError("No modalities provided to FusionCrossAttention")

        # helper to project or return zeros (B,1,fusion_dim)
        def proj_or_zero(x, proj):
            if x is None:
                return torch.zeros((B, 1, self.fusion_dim), device=device)
            else:
                return proj(x).unsqueeze(1)

        t = proj_or_zero(text_vec, self.pt)   # (B,1,fusion_dim)
        a = proj_or_zero(audio_vec, self.pa)
        v = proj_or_zero(visual_vec, self.pv)

        # For each modality as query, attend to the other modalities if present.
        # But keep fixed outputs for all three modalities (zero-filled if missing).
        # Text query
        t_klist = []
        if audio_vec is not None:
            t_klist.append(self.ta(t, a))
        if visual_vec is not None:
            t_klist.append(self.tv(t, v))
        if t_klist:
            t_out = torch.mean(torch.stack(t_klist, dim=0), dim=0).squeeze(1)
        else:
            t_out = t.squeeze(1)  # either projected text or zeros

        # Audio query
        a_klist = []
        if text_vec is not None:
            a_klist.append(self.at(a, t))
        if visual_vec is not None:
            a_klist.append(self.av(a, v))
        if a_klist:
            a_out = torch.mean(torch.stack(a_klist, dim=0), dim=0).squeeze(1)
        else:
            a_out = a.squeeze(1)

        # Visual query
        v_klist = []
        if text_vec is not None:
            v_klist.append(self.vt(v, t))
        if audio_vec is not None:
            v_klist.append(self.va(v, a))
        if v_klist:
            v_out = torch.mean(torch.stack(v_klist, dim=0), dim=0).squeeze(1)
        else:
            v_out = v.squeeze(1)

        # apply gates (broadcast)
        gated_t = t_out * self.gate[0]
        gated_a = a_out * self.gate[1]
        gated_v = v_out * self.gate[2]

        # always concatenate three parts -> fixed (B, fusion_dim*3)
        fused = torch.cat([gated_t, gated_a, gated_v], dim=1)
        return fused

class MFFNC(nn.Module):
    def __init__(self, text_dim=384, audio_dim=256, visual_dim=128, stats_dim=5, fusion_dim=256, hidden=256):
        """
        stats_dim default set to 5 to match the collate_fn that builds a 5-element stats vector.
        """
        super().__init__()
        # per-modality projectors (if already embeddings, you can fine-tune sizes)
        self.text_proj = nn.Linear(text_dim, 256)
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.visual_proj = nn.Linear(visual_dim, 256)
        # stats projector: input stats_dim -> 64
        self.stats_proj = MLP(stats_dim, 64, hidden=128)
        # fusion module (always returns fusion_dim * 3)
        self.fuser = FusionCrossAttention(dim_text=256, dim_audio=256, dim_visual=256, fusion_dim=fusion_dim)
        # combine fused + stats (fixed sizes: fusion_dim*3 + 64)
        fusion_out_dim = fusion_dim * 3
        self.comb = nn.Sequential(
            nn.Linear(fusion_out_dim + 64, hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # heads
        self.class_head = nn.Linear(hidden, 2)
        self.reg_head = nn.Linear(hidden, 1)

    def forward(self, text_emb=None, audio_emb=None, visual_emb=None, stats_vec=None):
        # project modality embeddings if present
        t = self.text_proj(text_emb) if text_emb is not None else None  # (B,256)
        a = self.audio_proj(audio_emb) if audio_emb is not None else None
        v = self.visual_proj(visual_emb) if visual_emb is not None else None

        # stats projection or zero (B,64)
        device = next(self.parameters()).device
        if stats_vec is not None:
            s = self.stats_proj(stats_vec)
        else:
            # infer batch size from any present modality
            bs = None
            if t is not None:
                bs = t.shape[0]
            elif a is not None:
                bs = a.shape[0]
            elif v is not None:
                bs = v.shape[0]
            else:
                raise ValueError("At least one modality must be provided")
            s = torch.zeros((bs, 64), device=device)

        # fused representation (always fixed size fusion_dim*3)
        fused = self.fuser(t, a, v)
        x = torch.cat([fused, s], dim=1)
        x = self.comb(x)
        logits = self.class_head(x)
        reg = self.reg_head(x).squeeze(1)
        return logits, reg
