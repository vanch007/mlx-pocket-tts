import mlx.core as mx

from mlx_pocket_tts.conversion import _reshape_for_target, map_weight_name


def test_official_weight_names_map_to_mlx_layout():
    assert map_weight_name("mimi.encoder.model.0.conv.weight") == (
        "mimi.encoder.init_conv1d.conv.conv.weight"
    )
    assert map_weight_name("mimi.decoder.model.2.convtr.weight") == (
        "mimi.decoder.layers.0.upsample.convtr.convtr.weight"
    )
    assert map_weight_name("flow_lm.speaker_proj_weight") == "speaker_proj_weight"


def test_convolution_weight_is_transposed_to_mlx_layout():
    source = mx.zeros((4, 3, 5))
    assert _reshape_for_target(source, (4, 5, 3), "conv").shape == (4, 5, 3)
