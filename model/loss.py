"""
Composite loss for rating prediction.

Combines three complementary objectives:
  - MSE: minimise per-sample prediction error.
  - PLCC (Pearson Linear Correlation Coefficient): maximise linear agreement
    between predicted and ground-truth score distributions within each batch.
  - Margin ranking: enforce correct relative ordering between sample pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionRatingLoss(nn.Module):
    """
    Weighted sum of MSE, 1-PLCC, and margin ranking losses.

    The PLCC and ranking terms act as distribution- and rank-aware regularisers
    that push the model beyond per-sample regression accuracy.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.mse_w = 1.0
        self.plcc_w = cfg["plcc_weight"]
        self.rank_w = 0.5
        self.rank_margin = 0.1

    def plcc_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """1 - Pearson correlation. Requires sufficient batch size for stability."""
        if preds.shape[0] < 8:
            return torch.tensor(0.0, device=preds.device)
        p_c = preds - preds.mean()
        t_c = targets - targets.mean()
        cov = (p_c * t_c).sum()
        p_std = torch.sqrt((p_c ** 2).sum() + 1e-8)
        t_std = torch.sqrt((t_c ** 2).sum() + 1e-8)
        return 1.0 - cov / (p_std * t_std)

    def margin_ranking_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss via random permutation."""
        n = preds.shape[0]
        if n < 4:
            return torch.tensor(0.0, device=preds.device)
        idx = torch.randperm(n, device=preds.device)
        p1, p2 = preds, preds[idx]
        t1, t2 = targets, targets[idx]
        y = torch.sign(t1 - t2)
        mask = y != 0
        if mask.sum() < 2:
            return torch.tensor(0.0, device=preds.device)
        return F.margin_ranking_loss(
            p1[mask], p2[mask], y[mask], margin=self.rank_margin
        )

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: scalar combined loss.
            preds: passthrough for convenience (e.g. logging).
        """
        mse = F.mse_loss(preds, targets)
        plcc = self.plcc_loss(preds, targets)
        rank = self.margin_ranking_loss(preds, targets)
        loss = self.mse_w * mse + self.plcc_w * plcc + self.rank_w * rank
        return loss, preds
