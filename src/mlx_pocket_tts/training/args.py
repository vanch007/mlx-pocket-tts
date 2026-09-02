from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlowArgs:
    type: str = "lsd"
    sig_min: float = 0.001
    p_equal: float = 0.75
    lognorm_mean: float = 0.4
    lognorm_std: float = 1.0
    normalize: bool = True
    stopgrad_type: str = "minimal"
    distill_prob: float = 0.25


@dataclass
class OptimArgs:
    lr: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_norm: float = 1.0
    warmup_steps: int = 500
    schedule: str = "constant"
    lr_min_ratio: float = 0.0


@dataclass
class TrainArgs:
    flow: FlowArgs = field(default_factory=FlowArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    flow_batch_multiplier: int = 1
    eos_loss_weight: float = 0.1
    text_dropout: float = 0.2
    voice_dropout: float = 0.2
    stats_ema_decay: float = 0.999
    ema_decay: float = 0.999
    distill_cfg_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.flow_batch_multiplier < 1:
            raise ValueError("flow_batch_multiplier must be positive")
        for name in ("text_dropout", "voice_dropout"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
