"""Tests for probes/get_probe.py factory dispatch and probe forward passes."""

import torch
import pytest

try:
    from src.protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
    from src.protify.probes.mlp_probe import MLPProbe
    from src.protify.probes.transformer_probe import (
        TransformerForSequenceClassification, TransformerForTokenClassification,
    )
    from src.protify.probes.lyra_probe import LyraForSequenceClassification, LyraForTokenClassification
except ImportError:
    try:
        from protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
        from protify.probes.mlp_probe import MLPProbe
        from protify.probes.transformer_probe import (
            TransformerForSequenceClassification, TransformerForTokenClassification,
        )
        from protify.probes.lyra_probe import LyraForSequenceClassification, LyraForTokenClassification
    except ImportError:
        from ..probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
        from ..probes.mlp_probe import MLPProbe
        from ..probes.transformer_probe import (
            TransformerForSequenceClassification, TransformerForTokenClassification,
        )
        from ..probes.lyra_probe import LyraForSequenceClassification, LyraForTokenClassification


def _make_args(**kwargs) -> ProbeArguments:
    defaults = dict(
        input_size=16,
        hidden_size=32,
        num_labels=3,
        n_layers=1,
        dropout=0.1,
        task_type="singlelabel",
        head_size=16,
        classifier_size=24,
        transformer_dropout=0.1,
        classifier_dropout=0.1,
        rotary=False,
        attention_backend="sdpa",
        output_s_max=False,
        pre_ln=True,
        probe_pooling_types=["mean"],
        use_bias=False,
        add_token_ids=False,
        expansion_ratio=8 / 3,
    )
    defaults.update(kwargs)
    return ProbeArguments(**defaults)


def test_get_probe_mlp_sequence() -> None:
    args = _make_args(probe_type="mlp", tokenwise=False)
    probe = get_probe(args)
    assert isinstance(probe, MLPProbe)


def test_probe_type_linear_still_selects_the_mlp_probe() -> None:
    # "linear" is the former name and appears in saved configs and existing YAML.
    args = _make_args(probe_type="linear", tokenwise=False)
    assert args.probe_type == "mlp"
    assert isinstance(get_probe(args), MLPProbe)


def test_get_probe_transformer_sequence() -> None:
    args = _make_args(probe_type="transformer", tokenwise=False)
    probe = get_probe(args)
    assert isinstance(probe, TransformerForSequenceClassification)


def test_get_probe_transformer_tokenwise() -> None:
    args = _make_args(probe_type="transformer", tokenwise=True)
    probe = get_probe(args)
    assert isinstance(probe, TransformerForTokenClassification)


def test_get_probe_lyra_sequence() -> None:
    args = _make_args(probe_type="lyra", tokenwise=False)
    probe = get_probe(args)
    assert isinstance(probe, LyraForSequenceClassification)


def test_get_probe_lyra_tokenwise() -> None:
    args = _make_args(probe_type="lyra", tokenwise=True)
    probe = get_probe(args)
    assert isinstance(probe, LyraForTokenClassification)


def test_get_probe_invalid_raises() -> None:
    args = _make_args(probe_type="nonexistent", tokenwise=False)
    with pytest.raises(ValueError):
        get_probe(args)


def test_linear_probe_forward_singlelabel() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="singlelabel", num_labels=3)
    probe = get_probe(args).eval()
    embeddings = torch.randn(2, 16)  # (b=2, d=16)
    output = probe(embeddings)  # logits: (b=2, c=3)
    assert output.logits.shape == (2, 3)


def test_linear_probe_forward_regression() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="regression", num_labels=1)
    probe = get_probe(args).eval()
    embeddings = torch.randn(2, 16)  # (b=2, d=16)
    output = probe(embeddings)  # logits: (b=2, c=1)
    assert output.logits.shape == (2, 1)


def test_linear_probe_forward_multilabel() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="multilabel", num_labels=4)
    probe = get_probe(args).eval()
    embeddings = torch.randn(2, 16)  # (b=2, d=16)
    output = probe(embeddings)  # logits: (b=2, c=4)
    assert output.logits.shape == (2, 4)


def test_linear_probe_forward_with_labels_returns_loss() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="singlelabel", num_labels=3)
    probe = get_probe(args)
    embeddings = torch.randn(2, 16)  # (b=2, d=16)
    labels = torch.tensor([0, 2])  # (b=2,)
    output = probe(embeddings, labels=labels)  # logits: (b=2, c=3); loss: ()
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert output.loss.item() > 0


def test_linear_probe_forward_sigmoid_regression() -> None:
    torch.manual_seed(0)
    args = _make_args(
        probe_type="linear", tokenwise=False,
        task_type="sigmoid_regression", num_labels=1,
    )
    probe = get_probe(args).eval()
    embeddings = torch.randn(2, 16)  # (b=2, d=16)
    output = probe(embeddings)  # logits: (b=2, c=1)
    assert output.logits.shape == (2, 1)
    # sigmoid_regression should output values in [0, 1]
    assert (output.logits >= 0).all()  # (b, c) -> ()
    assert (output.logits <= 1).all()  # (b, c) -> ()


def test_rebuild_probe_from_saved_config_linear() -> None:
    args = _make_args(probe_type="linear", tokenwise=False, num_labels=3)
    original = get_probe(args)
    config_dict = original.config.to_dict()
    rebuilt = rebuild_probe_from_saved_config("linear", False, config_dict)
    assert isinstance(rebuilt, MLPProbe)
    assert rebuilt.config.num_labels == 3


def test_transformer_probe_config_legacy_n_heads_kwarg() -> None:
    """Back-compat: checkpoints serialized with `n_heads` must still load."""
    try:
        from src.protify.probes.transformer_probe import TransformerProbeConfig
    except ImportError:
        try:
            from protify.probes.transformer_probe import TransformerProbeConfig
        except ImportError:
            from ..probes.transformer_probe import TransformerProbeConfig
    cfg = TransformerProbeConfig(hidden_size=512, n_heads=4)
    assert cfg.head_size == 128
    assert cfg.n_heads == 4


def test_transformer_probe_config_head_size_and_legacy_agree() -> None:
    """Both head_size and legacy n_heads provided, consistent values."""
    try:
        from src.protify.probes.transformer_probe import TransformerProbeConfig
    except ImportError:
        try:
            from protify.probes.transformer_probe import TransformerProbeConfig
        except ImportError:
            from ..probes.transformer_probe import TransformerProbeConfig
    cfg = TransformerProbeConfig(hidden_size=512, head_size=128, n_heads=4)
    assert cfg.head_size == 128
    assert cfg.n_heads == 4


def test_transformer_probe_config_head_size_and_legacy_conflict() -> None:
    try:
        from src.protify.probes.transformer_probe import TransformerProbeConfig
    except ImportError:
        try:
            from protify.probes.transformer_probe import TransformerProbeConfig
        except ImportError:
            from ..probes.transformer_probe import TransformerProbeConfig
    with pytest.raises(AssertionError):
        TransformerProbeConfig(hidden_size=512, head_size=64, n_heads=4)


def test_transformer_probe_config_rebuild_from_legacy_dict() -> None:
    """A saved config.json with only `n_heads` must roundtrip through rebuild_probe_from_saved_config."""
    legacy_dict = {
        "input_size": 16,
        "hidden_size": 32,
        "classifier_size": 24,
        "transformer_dropout": 0.1,
        "classifier_dropout": 0.1,
        "num_labels": 2,
        "n_layers": 1,
        "n_heads": 2,  # legacy key only
        "task_type": "singlelabel",
        "rotary": False,
        "attention_backend": "sdpa",
        "pre_ln": True,
        "pooling_types": ["mean"],
        "use_bias": False,
        "add_token_ids": False,
    }
    rebuilt = rebuild_probe_from_saved_config("transformer", False, legacy_dict)
    assert isinstance(rebuilt, TransformerForSequenceClassification)
    assert rebuilt.config.head_size == 16
    assert rebuilt.config.n_heads == 2
