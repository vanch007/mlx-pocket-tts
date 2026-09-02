import mlx.core as mx
import mlx.nn as nn

from mlx_pocket_tts.conditioners import TokenizedText
from mlx_pocket_tts.training import (
    EMA,
    Batch,
    OptimArgs,
    Trainer,
    build_optimizer,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from mlx_pocket_tts.training.objectives import LSD


class ObjectiveFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(6, 2)
        self.objective = LSD(distill_prob=1, stopgrad_type="minimal")

    def __call__(self, conditioning, noise, target):
        def flow(s, t, value):
            return self.head(mx.concatenate((conditioning, s, t, value), axis=-1))

        loss, _, _ = self.objective(flow, noise, target)
        return mx.mean(loss)


def test_lsd_minimal_path_has_finite_gradients():
    model = ObjectiveFixture()
    conditioning = mx.ones((3, 2))
    noise = mx.zeros((3, 2))
    target = mx.ones((3, 2))
    loss, gradients = nn.value_and_grad(model, lambda module: module(conditioning, noise, target))(
        model
    )
    mx.eval(loss, gradients)
    assert bool(mx.isfinite(loss).item())
    assert bool(mx.all(mx.isfinite(gradients["head"]["weight"])).item())
    assert float(mx.sum(mx.abs(gradients["head"]["weight"])).item()) > 0


def test_flow_matching_compatibility_objective_has_finite_gradients():
    from mlx_pocket_tts.training import FlowMatching

    model = ObjectiveFixture()
    objective = FlowMatching()
    conditioning = mx.ones((3, 2))
    noise = mx.zeros((3, 2))
    target = mx.ones((3, 2))

    def loss_fn(module):
        values, _, _ = objective(
            lambda s, t, value: module.head(mx.concatenate((conditioning, s, t, value), axis=-1)),
            noise,
            target,
        )
        return mx.mean(values)

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, gradients)
    assert bool(mx.isfinite(loss).item())
    assert float(mx.sum(mx.abs(gradients["head"]["weight"])).item()) > 0


class LinearLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def __call__(self, batch, *, update_stats=False):
        prediction = self.linear(batch)
        loss = mx.mean(mx.square(prediction - 2 * batch))
        return loss, {"loss": loss}


def test_atomic_checkpoint_restores_model_optimizer_and_ema(tmp_path):
    model = LinearLoss()
    optimizer = build_optimizer(model, OptimArgs(lr=0.05, weight_decay=0))
    ema = EMA(model, 0.9)
    trainer = Trainer(model, optimizer, ema=ema)
    trainer.step(mx.array([[1.0], [2.0]]))
    expected_weight = mx.array(model.linear.weight)
    expected_step = int(optimizer.step.item())
    checkpoint = save_checkpoint(tmp_path, 1, model, optimizer, ema)

    trainer.step(mx.array([[3.0]]))
    restored_step = load_checkpoint(checkpoint, model, optimizer, ema)
    assert restored_step == 1
    assert int(optimizer.step.item()) == expected_step
    assert mx.allclose(model.linear.weight, expected_weight)
    assert latest_checkpoint(tmp_path) == checkpoint


class TinyConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def __call__(self, value: TokenizedText):
        return self.embed(value.tokens)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def __call__(self, value, cache=None):
        return self.linear(value)


class TinyFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 2)

    def __call__(self, conditioning, s, t, value):
        return self.linear(mx.concatenate((conditioning, s, t, value), axis=-1))


class TinyFlowLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.ldim = 2
        self.emb_mean = mx.zeros((2,))
        self.emb_std = mx.ones((2,))
        self.bos_emb = mx.zeros((2,))
        self.bos_before_voice = mx.zeros((1, 1, 4))
        self.input_linear = nn.Linear(2, 4, bias=False)
        self.conditioner = TinyConditioner()
        self.transformer = TinyTransformer()
        self.out_norm = nn.LayerNorm(4)
        self.out_eos = nn.Linear(4, 1)
        self.flow_net = TinyFlowNet()


class TinyTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_lm = TinyFlowLM()
        self.speaker_proj_weight = mx.ones((4, 2))


def test_trainable_tts_runs_native_optimizer_step():
    from mlx_pocket_tts.training import FlowArgs, TrainableTTS, TrainArgs

    model = TrainableTTS(
        TinyTTS(), TrainArgs(flow=FlowArgs(distill_prob=1), text_dropout=0, voice_dropout=0)
    )
    optimizer = build_optimizer(model, OptimArgs(lr=0.01, weight_decay=0))
    trainer = Trainer(model, optimizer)
    batch = Batch(
        latents=mx.array([[[0.1, -0.1], [0.2, -0.2], [0.0, 0.0]]]),
        mask=mx.array([[True, True, False]]),
        text_tokens=[mx.array([1, 2], dtype=mx.int32)],
        voice_latents=mx.array([[[0.3, -0.3]]]),
        num_voice_prompt_frames=mx.array([1], dtype=mx.int32),
    )
    result = trainer.step(batch)
    assert result["loss"] > 0
    assert result["grad_norm"] > 0
    assert int(optimizer.step.item()) == 1


def test_cfg_distillation_freezes_teacher_and_heads():
    from mlx.utils import tree_flatten

    from mlx_pocket_tts.training import (
        FlowArgs,
        TrainableTTS,
        TrainArgs,
        attach_distillation,
    )

    teacher_base = TinyTTS()
    student = TrainableTTS(
        TinyTTS(), TrainArgs(flow=FlowArgs(distill_prob=1), text_dropout=0, voice_dropout=0)
    )
    selected = attach_distillation(student, teacher_base, cfg_coef=1)
    batch = Batch(
        latents=mx.array([[[0.1, -0.1], [0.2, -0.2], [0.0, 0.0]]]),
        mask=mx.array([[True, True, False]]),
        text_tokens=[mx.array([1, 2], dtype=mx.int32)],
        voice_latents=mx.array([[[0.3, -0.3]]]),
        num_voice_prompt_frames=mx.array([1], dtype=mx.int32),
    )
    loss, metrics = student(batch)
    mx.eval(loss)
    trainable_names = [name for name, _ in tree_flatten(student.trainable_parameters())]
    assert selected is None
    assert "distill_mse" in metrics
    assert not any("distill_teacher" in name for name in trainable_names)
    assert not any(name.startswith("flow_lm.flow_net") for name in trainable_names)
    assert not any(name.startswith("flow_lm.out_eos") for name in trainable_names)


def test_depth_distillation_selects_teacher_ends():
    from mlx_pocket_tts.training import select_teacher_layers

    assert select_teacher_layers(24, 6) == [0, 1, 2, 21, 22, 23]
