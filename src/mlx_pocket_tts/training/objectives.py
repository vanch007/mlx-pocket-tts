from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

FlowNet = Callable[[mx.array, mx.array, mx.array], mx.array]


class FlowMatching(nn.Module):
    """Optimal-transport conditional flow matching compatibility objective."""

    def __init__(self, sig_min: float = 0.001):
        super().__init__()
        self.sig_min = sig_min

    def __call__(
        self, flow: FlowNet, noise: mx.array, target: mx.array
    ) -> tuple[mx.array, dict[str, mx.array], mx.array]:
        t = mx.random.uniform(shape=target[..., :1].shape)
        x_t = (1 - (1 - self.sig_min) * t) * noise + t * target
        direction = target - (1 - self.sig_min) * noise
        loss = mx.mean(mx.square(flow(t, t, x_t) - direction), axis=-1)
        return loss, {"flow_matching": mx.mean(loss)}, t


class WeightMLP(nn.Module):
    def __init__(self, width: int = 32, depth: int = 3):
        super().__init__()
        self.layers = []
        input_width = 2
        for _ in range(depth):
            self.layers.extend((nn.Linear(input_width, width), nn.ReLU()))
            input_width = width
        self.layers.append(nn.Linear(input_width, 1))
        final = self.layers[-1]
        final.weight = mx.zeros_like(final.weight)
        final.bias = mx.zeros_like(final.bias)

    def __call__(self, s: mx.array, t: mx.array) -> mx.array:
        value = mx.concatenate((s, t), axis=-1)
        for layer in self.layers:
            value = layer(value)
        return value


def _input_gradient_only(function: Callable[[mx.array], mx.array], value: mx.array) -> mx.array:
    # Captured arrays are constants to an MLX custom function. The default VJP
    # therefore propagates through `value`, but not through the captured module weights.
    @mx.custom_function
    def apply(input_value: mx.array) -> mx.array:
        return function(input_value)

    return apply(value)


class LSD(nn.Module):
    """Native MLX Lagrangian Self-Distillation objective used by Pocket TTS."""

    def __init__(
        self,
        *,
        p_equal: float = 0.75,
        lognorm_mean: float = 0.4,
        lognorm_std: float = 1.0,
        normalize: bool = True,
        stopgrad_type: str = "minimal",
        distill_prob: float = 0.25,
    ):
        super().__init__()
        if not 0 < distill_prob <= 1:
            raise ValueError("distill_prob must be in (0, 1]")
        if stopgrad_type not in {"minimal", "classic"}:
            raise ValueError("stopgrad_type must be 'minimal' or 'classic'")
        self.p_equal = p_equal
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.normalize = normalize
        self.stopgrad_type = stopgrad_type
        self.distill_prob = distill_prob
        if normalize:
            self.w_s_t = WeightMLP()

    def _sample_t(self, value: mx.array) -> mx.array:
        logits = mx.random.normal(value[..., :1].shape) * self.lognorm_std + self.lognorm_mean
        return mx.sigmoid(logits)

    def _sample_s_t(self, value: mx.array) -> tuple[mx.array, mx.array]:
        logits = mx.random.normal(value[..., :2].shape) * self.lognorm_std + self.lognorm_mean
        probabilities = mx.sigmoid(logits)
        return mx.min(probabilities, axis=-1, keepdims=True), mx.max(
            probabilities, axis=-1, keepdims=True
        )

    def __call__(
        self, flow: FlowNet, noise: mx.array, target: mx.array
    ) -> tuple[mx.array, dict[str, mx.array], mx.array]:
        t = self._sample_t(target)
        x_t = t * target + (1 - t) * noise
        direction = target - noise
        diagonal = mx.sum(mx.square(flow(t, t, x_t) - direction), axis=-1)
        metrics = {"flow_diag": mx.mean(diagonal)}
        if self.normalize:
            logvar = self.w_s_t(t, t).squeeze(-1)
            diagonal = diagonal * mx.exp(logvar) / noise.shape[-1] - logvar

        skip = (
            self.training
            and self.distill_prob < 1
            and bool(mx.random.uniform(shape=()).item() >= self.distill_prob)
        )
        if skip:
            return self.p_equal * diagonal, metrics, t

        s, endpoint = self._sample_s_t(target)
        x_s = s * target + (1 - s) * noise

        def along_t(t_value: mx.array) -> mx.array:
            return flow(s, t_value, x_s)

        (velocity,), (dvdt,) = mx.jvp(along_t, [endpoint], [mx.ones_like(endpoint)])
        x_endpoint = x_s + (endpoint - s) * velocity
        dxdt = velocity + (endpoint - s) * dvdt
        if self.stopgrad_type == "minimal":
            target_velocity = _input_gradient_only(
                lambda value: flow(endpoint, endpoint, value), x_endpoint
            )
        else:
            target_velocity = mx.stop_gradient(flow(endpoint, endpoint, x_endpoint))
        distill = mx.sum(mx.square(dxdt - target_velocity), axis=-1)
        metrics["flow_distill"] = mx.mean(distill)
        if self.normalize:
            logvar = self.w_s_t(s, endpoint).squeeze(-1)
            distill = distill * mx.exp(logvar) / noise.shape[-1] - logvar
        weight = (1 - self.p_equal) / (self.distill_prob if self.training else 1)
        return self.p_equal * diagonal + weight * distill, metrics, t
