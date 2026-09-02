from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_pocket_tts.conditioners import TokenizedText

from .args import TrainArgs
from .batch import Batch
from .objectives import LSD, FlowMatching


def _bernoulli_keep(batch_size: int, probability: float) -> list[bool]:
    if probability <= 0:
        return [False] * batch_size
    if probability >= 1:
        return [True] * batch_size
    return [
        bool(value) for value in (mx.random.uniform(shape=(batch_size,)) < probability).tolist()
    ]


class TrainableTTS(nn.Module):
    """Trainable MLX wrapper whose FlowLM parameters remain inference-compatible."""

    def __init__(self, model, args: TrainArgs | None = None):
        super().__init__()
        self.args = args or TrainArgs()
        self.flow_lm = model.flow_lm
        self.speaker_proj_weight = model.speaker_proj_weight
        flow_args = vars(self.args.flow).copy()
        flow_type = flow_args.pop("type")
        sig_min = flow_args.pop("sig_min")
        if flow_type == "lsd":
            self.flow = LSD(**flow_args)
        elif flow_type == "flow_matching":
            self.flow = FlowMatching(sig_min=sig_min)
        else:
            raise ValueError(f"Unknown flow objective {flow_type!r}")
        self.flow_lm.freeze(recurse=False, keys=["emb_mean", "emb_std"])
        self.__dict__["distill_teacher_flow_lm"] = None
        self.__dict__["distill_teacher_speaker_proj_weight"] = None

    def _update_latent_stats(self, latents: mx.array, mask: mx.array) -> None:
        selected = latents.reshape(-1, latents.shape[-1])[mask.reshape(-1)]
        decay = self.args.stats_ema_decay
        self.flow_lm.emb_mean = decay * self.flow_lm.emb_mean + (1 - decay) * mx.mean(
            selected, axis=0
        )
        self.flow_lm.emb_std = decay * self.flow_lm.emb_std + (1 - decay) * mx.std(
            selected, axis=0, ddof=1
        )

    def _backbone(
        self,
        normalized: mx.array,
        text_tokens: list[mx.array],
        voice_latents: mx.array,
        prompt_lengths: mx.array,
        *,
        cfg_dropout: bool,
        force_null: bool = False,
        flow_lm=None,
        speaker_proj_weight: mx.array | None = None,
    ) -> mx.array:
        fl = flow_lm or self.flow_lm
        projection = (
            self.speaker_proj_weight if speaker_proj_weight is None else speaker_proj_weight
        )
        batch_size, frames, _ = normalized.shape
        if force_null:
            keep_voice = keep_text = [False] * batch_size
        elif cfg_dropout:
            keep_voice = _bernoulli_keep(batch_size, 1 - self.args.voice_dropout)
            keep_text = _bernoulli_keep(batch_size, 1 - self.args.text_dropout)
        else:
            keep_voice = keep_text = [True] * batch_size

        bos_audio = mx.broadcast_to(fl.bos_emb[None, None], (batch_size, 1, fl.ldim))
        audio_input = mx.concatenate((bos_audio, normalized[:, :-1]), axis=1)
        audio_embeddings = fl.input_linear(audio_input)
        voice_embeddings = voice_latents @ projection.T
        rows = []
        prefix_lengths = []
        for index, tokens in enumerate(text_tokens):
            parts = [fl.bos_before_voice[0]]
            if keep_voice[index]:
                count = min(int(prompt_lengths[index].item()), voice_embeddings.shape[1])
                parts.append(voice_embeddings[index, :count])
            if keep_text[index]:
                parts.append(fl.conditioner(TokenizedText(tokens[None]))[0])
            prefix_lengths.append(sum(part.shape[0] for part in parts))
            parts.append(audio_embeddings[index])
            rows.append(mx.concatenate(parts, axis=0))
        width = max(row.shape[0] for row in rows)
        sequence = mx.stack(
            [mx.pad(row, ((0, width - row.shape[0]), (0, 0))) for row in rows], axis=0
        )
        transformed = fl.out_norm(fl.transformer(sequence, cache=None))
        return mx.stack(
            [
                transformed[index, prefix : prefix + frames]
                for index, prefix in enumerate(prefix_lengths)
            ]
        )

    def __call__(self, batch: Batch, *, update_stats: bool = False):
        latents, mask = batch.latents, batch.mask
        if update_stats and self.training:
            self._update_latent_stats(latents, mask)
        mean = mx.stop_gradient(self.flow_lm.emb_mean)
        std = mx.stop_gradient(self.flow_lm.emb_std)
        normalized = (latents - mean) / std
        hidden = self._backbone(
            normalized,
            batch.text_tokens,
            batch.voice_latents,
            batch.num_voice_prompt_frames,
            cfg_dropout=self.training,
        )

        if self.distill_teacher_flow_lm is not None:
            teacher = self.distill_teacher_flow_lm
            teacher_projection = self.distill_teacher_speaker_proj_weight
            conditioned = self._backbone(
                normalized,
                batch.text_tokens,
                batch.voice_latents,
                batch.num_voice_prompt_frames,
                cfg_dropout=False,
                flow_lm=teacher,
                speaker_proj_weight=teacher_projection,
            )
            null = self._backbone(
                normalized,
                batch.text_tokens,
                batch.voice_latents,
                batch.num_voice_prompt_frames,
                cfg_dropout=False,
                force_null=True,
                flow_lm=teacher,
                speaker_proj_weight=teacher_projection,
            )
            target_hidden = mx.stop_gradient(
                null + self.args.distill_cfg_coef * (conditioned - null)
            )
            shifted_mask = mx.concatenate((mask[:, :1], mask[:, :-1]), axis=1)
            mse = mx.mean(mx.square(hidden - target_hidden), axis=-1)
            loss = mx.sum(mse * shifted_mask) / mx.maximum(mx.sum(shifted_mask), mx.array(1))
            return loss, {"distill_mse": loss, "loss": loss}

        eos = ~mask
        eos = mx.concatenate((mx.zeros_like(eos[:, :1]), eos[:, 1:]), axis=1)
        logits = self.flow_lm.out_eos(hidden).squeeze(-1)
        shifted_mask = mx.concatenate((mask[:, :1], mask[:, :-1]), axis=1)
        eos_loss_values = mx.where(
            eos,
            mx.logaddexp(mx.zeros_like(logits), -logits),
            mx.logaddexp(mx.zeros_like(logits), logits),
        )
        denominator = mx.maximum(mx.sum(shifted_mask), mx.array(1))
        eos_loss = mx.sum(eos_loss_values * shifted_mask) / denominator

        multiplier = self.args.flow_batch_multiplier
        flow_hidden = mx.broadcast_to(hidden[None], (multiplier,) + hidden.shape).reshape(
            -1, hidden.shape[-1]
        )
        targets = mx.broadcast_to(normalized[None], (multiplier,) + normalized.shape).reshape(
            -1, normalized.shape[-1]
        )
        flow_mask = mx.broadcast_to(mask[None], (multiplier,) + mask.shape).reshape(-1)
        noise = mx.random.normal(targets.shape, dtype=targets.dtype)
        losses, metrics, _ = self.flow(
            lambda s, t, value: self.flow_lm.flow_net(flow_hidden, s, t, value), noise, targets
        )
        flow_loss = mx.sum(losses * flow_mask) / mx.maximum(mx.sum(flow_mask), mx.array(1))
        loss = flow_loss + self.args.eos_loss_weight * eos_loss
        metrics.update(flow_loss=flow_loss, eos_loss=eos_loss, loss=loss)
        return loss, metrics
