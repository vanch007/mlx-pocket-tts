from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from .args import OptimArgs
from .batch import Batch
from .checkpoint import EMA


def build_optimizer(model, args: OptimArgs):
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=list(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        bias_correction=True,
    )
    optimizer.init(model.trainable_parameters())
    return optimizer


class Trainer:
    def __init__(self, model, optimizer, *, max_norm: float = 1.0, ema: EMA | None = None):
        self.model = model
        self.optimizer = optimizer
        self.max_norm = max_norm
        self.ema = ema
        self._value_and_grad = nn.value_and_grad(model, self._loss)
        self._active_batch: Batch | None = None
        self._active_update_stats = False

    def _loss(self, model):
        if self._active_batch is None:
            raise RuntimeError("No active training batch")
        loss, _ = model(self._active_batch, update_stats=self._active_update_stats)
        return loss

    def step(
        self, batches: Batch | Iterable[Batch], *, update_stats: bool = False
    ) -> dict[str, float]:
        if isinstance(batches, (list, tuple)):
            micro_batches = list(batches)
        else:
            micro_batches = [batches]
        if not micro_batches:
            raise ValueError("At least one micro-batch is required")
        accumulated = None
        losses = []
        for batch in micro_batches:
            self._active_batch = batch
            self._active_update_stats = update_stats
            loss, gradients = self._value_and_grad(self.model)
            losses.append(loss)
            if accumulated is None:
                accumulated = gradients
            else:
                accumulated = tree_map(lambda left, right: left + right, accumulated, gradients)
        scale = 1 / len(micro_batches)
        accumulated = tree_map(lambda value: value * scale, accumulated)
        clipped, grad_norm = optim.clip_grad_norm(accumulated, self.max_norm)
        if not bool(mx.isfinite(grad_norm).item()):
            raise FloatingPointError("Non-finite gradient norm")
        self.optimizer.update(self.model, clipped)
        if self.ema is not None:
            self.ema.update(self.model)
        mean_loss = mx.mean(mx.stack(losses))
        mx.eval(self.model.parameters(), self.optimizer.state, mean_loss, grad_norm)
        self._active_batch = None
        return {"loss": float(mean_loss.item()), "grad_norm": float(grad_norm.item())}
