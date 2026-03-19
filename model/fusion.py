"""
SigLIP2-based multimodal fusion model for product rating prediction.

Architecture overview:
  1. Dual-stream encoding via a frozen SigLIP2 backbone (vision + text).
     Last N vision blocks are optionally unfrozen for fine-tuning.
  2. Modality-specific linear projections into a shared fusion space.
  3. Learnable query tokens + type embeddings enable cross-modal interaction
     through a lightweight Transformer fusion stack.
  4. Gated pooling combines image and text representations.
  5. A multi-layer regression head maps the fused features (plus raw cosine
     similarity) to a bounded score in [1, 5].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock(nn.Module):
    """Single Transformer block for self-attention over multimodal token sequences."""

    def __init__(self, d: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x


class SigLIP2FusionModel(nn.Module):
    """
    End-to-end multimodal model: SigLIP2 encoder → fusion Transformer → rating.
    """

    def __init__(self, clip_model: nn.Module, cfg: dict):
        super().__init__()
        d = cfg["embed_dim"]       # 768 (SigLIP2 output dim)
        fd = 512                   # fusion dimension
        drop = cfg["dropout"]
        n_queries = 4

        # ── Encoder (partially frozen) ──
        self.clip = clip_model
        self._setup_freezing(cfg)

        # ── Projection from encoder dim to fusion dim ──
        self.img_proj = nn.Sequential(nn.Linear(d, fd), nn.LayerNorm(fd))
        self.txt_proj = nn.Sequential(nn.Linear(d, fd), nn.LayerNorm(fd))

        # ── Type embeddings + learnable query tokens ──
        self.img_type = nn.Parameter(torch.randn(1, 1, fd) * 0.02)
        self.txt_type = nn.Parameter(torch.randn(1, 1, fd) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, n_queries, fd) * 0.02)

        # ── Fusion: 2-layer Transformer over [img, txt, queries] ──
        self.fusion = nn.ModuleList([
            FusionBlock(fd, num_heads=8, dropout=drop)
            for _ in range(2)
        ])

        # ── Gated pooling: soft routing between image and text streams ──
        self.gate = nn.Sequential(nn.Linear(fd * 2, fd), nn.Sigmoid())

        # ── Output head: fused features (fd) + cosine similarity (1) → score ──
        self.head = nn.Sequential(
            nn.LayerNorm(fd + 1),
            nn.Linear(fd + 1, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop),
            nn.Linear(64, 1),
        )

        self._print_params()

    # ── Freezing strategy ──

    def _setup_freezing(self, cfg: dict) -> None:
        """Freeze the encoder, then selectively unfreeze the last N vision blocks."""
        for p in self.clip.parameters():
            p.requires_grad = False

        n = cfg.get("unfreeze_vision_layers", 2)

        # Locate vision blocks (handles multiple open_clip structures)
        blocks = None
        for attr_path in ["visual.trunk.blocks", "visual.blocks"]:
            obj = self.clip
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                blocks = list(obj)
                break
            except AttributeError:
                continue

        if blocks is not None:
            for i in range(max(0, len(blocks) - n), len(blocks)):
                for p in blocks[i].parameters():
                    p.requires_grad = True

            # Also unfreeze final layer norm
            for attr_path in ["visual.trunk.norm", "visual.norm"]:
                obj = self.clip
                try:
                    for attr in attr_path.split("."):
                        obj = getattr(obj, attr)
                    for p in obj.parameters():
                        p.requires_grad = True
                    break
                except AttributeError:
                    continue

            v_train = sum(
                p.numel() for p in self.clip.visual.parameters() if p.requires_grad
            )
            print(f"[VISION] {len(blocks)} blocks, unfroze last {n}: {v_train / 1e6:.2f}M trainable")
        else:
            print("[VISION] Could not find blocks — fully frozen")

        t_total = sum(p.numel() for p in self.clip.parameters()) - sum(
            p.numel() for p in self.clip.visual.parameters()
        )
        print(f"[TEXT]   FULLY FROZEN: {t_total / 1e6:.1f}M")

    def _print_params(self) -> None:
        """Log parameter counts for the encoder and fusion+head."""
        clip_total = sum(p.numel() for p in self.clip.parameters())
        clip_train = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        head_params = sum(
            p.numel()
            for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("clip.")
        )
        total_train = clip_train + head_params
        print(f"\n[MODEL] Encoder: {clip_total / 1e6:.1f}M ({clip_train / 1e6:.2f}M trainable)")
        print(f"[MODEL] Fusion+Head: {head_params / 1e6:.2f}M (trainable)")
        print(f"[MODEL] Total trainable: {total_train / 1e6:.2f}M")

    # ── Forward pass ──

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) preprocessed image batch.
            tokens: (B, context_length) tokenised text batch.

        Returns:
            scores: (B,) predicted ratings in [1, 5].
        """
        # Encode (gradients flow through unfrozen vision blocks only)
        img_emb = self.clip.encode_image(images, normalize=True)       # (B, 768)
        with torch.no_grad():
            txt_emb = self.clip.encode_text(tokens, normalize=True)    # (B, 768)
        txt_emb = txt_emb.detach()

        # Raw cosine similarity as an auxiliary signal
        cos = F.cosine_similarity(img_emb, txt_emb, dim=-1).unsqueeze(-1)  # (B, 1)

        # Project to fusion dimension and add type embeddings
        img_tok = self.img_proj(img_emb).unsqueeze(1) + self.img_type  # (B, 1, fd)
        txt_tok = self.txt_proj(txt_emb).unsqueeze(1) + self.txt_type  # (B, 1, fd)
        q_tok = self.queries.expand(images.shape[0], -1, -1)           # (B, Q, fd)

        # Fusion: self-attention over [img_token, txt_token, query_tokens]
        seq = torch.cat([img_tok, txt_tok, q_tok], dim=1)  # (B, 2+Q, fd)
        for block in self.fusion:
            seq = block(seq)

        # Gated pooling: learn to weight image vs text contribution
        img_out = seq[:, 0, :]            # (B, fd)
        txt_out = seq[:, 1, :]            # (B, fd)
        query_out = seq[:, 2:, :].mean(1) # (B, fd)

        g = self.gate(torch.cat([img_out, txt_out], dim=-1))    # (B, fd)
        fused = g * img_out + (1 - g) * txt_out + query_out     # (B, fd)

        # Final score: sigmoid → [1, 5] range
        features = torch.cat([fused, cos], dim=-1)  # (B, fd+1)
        score = 1.0 + 4.0 * torch.sigmoid(self.head(features).squeeze(-1))  # (B,)

        return score
