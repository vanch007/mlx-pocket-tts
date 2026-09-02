from __future__ import annotations

import re

from mlx.utils import tree_flatten


def select_teacher_layers(n_teacher: int, n_student: int) -> list[int]:
    """Select the bottom and top teacher layers, matching the official recipe."""
    if not 0 < n_student <= n_teacher:
        raise ValueError(f"Cannot shrink teacher depth {n_teacher} to {n_student}")
    head = (n_student + 1) // 2
    tail = n_student - head
    return list(range(head)) + list(range(n_teacher - tail, n_teacher))


def seed_student_from_teacher(student, teacher) -> list[int]:
    student_depth = len(student.transformer.layers)
    teacher_depth = len(teacher.transformer.layers)
    selected = select_teacher_layers(teacher_depth, student_depth)
    remap = {source: target for target, source in enumerate(selected)}
    student_shapes = {
        name: tuple(value.shape) for name, value in tree_flatten(student.parameters())
    }
    copied = {}
    pattern = re.compile(r"^transformer\.layers\.(\d+)\.(.*)$")
    for name, value in tree_flatten(teacher.parameters()):
        match = pattern.match(name)
        destination = name
        if match:
            source_layer = int(match.group(1))
            if source_layer not in remap:
                continue
            destination = f"transformer.layers.{remap[source_layer]}.{match.group(2)}"
        if destination in student_shapes and student_shapes[destination] == tuple(value.shape):
            copied[destination] = value
    student.load_weights(list(copied.items()), strict=False)
    return selected


def attach_distillation(
    student,
    teacher_model,
    *,
    cfg_coef: float,
    seed_from_teacher: bool = False,
) -> list[int] | None:
    if cfg_coef <= 0:
        raise ValueError("cfg_coef must be positive")
    selected = None
    if seed_from_teacher:
        selected = seed_student_from_teacher(student.flow_lm, teacher_model.flow_lm)
    teacher_model.flow_lm.freeze()
    student.flow_lm.flow_net.freeze()
    student.flow_lm.out_eos.freeze()
    student.flow.freeze()
    student.args.distill_cfg_coef = cfg_coef
    # Direct dictionary insertion keeps the frozen teacher outside the student's
    # parameter tree and therefore outside checkpoints and optimizer state.
    student.__dict__["distill_teacher_flow_lm"] = teacher_model.flow_lm
    student.__dict__["distill_teacher_speaker_proj_weight"] = teacher_model.speaker_proj_weight
    return selected
