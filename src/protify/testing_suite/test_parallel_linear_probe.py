import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

try:
    from src.protify import probes as probes_package
    from src.protify.probes import trainers as trainers_module
    from src.protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
    from src.protify.probes.parallel_linear_probe import (
        ParallelLinearProbe,
        ParallelLinearProbeConfig,
        ParallelLinearProbeEnsemble,
    )
    from src.protify.probes.parallel_probe_batches import ParallelRunDataset
    from src.protify.probes.trainers import (
        ParallelProbePerRunGradientClipCallback,
        TrainerArguments,
        TrainerMixin,
        _clip_parallel_probe_gradients_per_run,
        _compute_eval_accumulation_steps,
    )
except ImportError:
    try:
        from protify import probes as probes_package
        from protify.probes import trainers as trainers_module
        from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
        from protify.probes.parallel_linear_probe import (
            ParallelLinearProbe,
            ParallelLinearProbeConfig,
            ParallelLinearProbeEnsemble,
        )
        from protify.probes.parallel_probe_batches import ParallelRunDataset
        from protify.probes.trainers import (
            ParallelProbePerRunGradientClipCallback,
            TrainerArguments,
            TrainerMixin,
            _clip_parallel_probe_gradients_per_run,
            _compute_eval_accumulation_steps,
        )
    except ImportError:
        from .. import probes as probes_package
        from ..probes import trainers as trainers_module
        from ..probes.mlp_probe import MLPProbe, MLPProbeConfig
        from ..probes.parallel_linear_probe import (
            ParallelLinearProbe,
            ParallelLinearProbeConfig,
            ParallelLinearProbeEnsemble,
        )
        from ..probes.parallel_probe_batches import ParallelRunDataset
        from ..probes.trainers import (
            ParallelProbePerRunGradientClipCallback,
            TrainerArguments,
            TrainerMixin,
            _clip_parallel_probe_gradients_per_run,
            _compute_eval_accumulation_steps,
        )


def _trainer_mixin_for_parallel_tests(task_type: str = 'singlelabel') -> TrainerMixin:
    trainer_args = TrainerArguments(
        model_save_dir='parallel-test',
        task_type=task_type,
        num_runs=2,
        parallel_probe_runs=True,
        balanced_regression_metrics=False,
        torch_compile=False,
    )
    mixin = TrainerMixin(trainer_args=trainer_args)
    mixin.embedding_args = SimpleNamespace(matrix_embed=False)
    mixin.probe_args = SimpleNamespace(
        probe_type='mlp',
        tokenwise=False,
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type=task_type,
        pre_ln=True,
        use_bias=True,
    )
    return mixin


def test_parallel_linear_probe_forward_returns_run_dimension() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=5,
        run_seeds=[11, 12, 13, 14, 15],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(7, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])

    output = model(embeddings=embeddings, labels=labels)

    assert output.logits.shape == (7, 5, 3)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_parallel_linear_probe_config_rejects_invalid_run_counts() -> None:
    with pytest.raises(AssertionError, match="num_runs must be positive"):
        ParallelLinearProbeConfig(num_runs=0)

    with pytest.raises(AssertionError, match="run_seeds length must match num_runs"):
        ParallelLinearProbeConfig(num_runs=2, run_seeds=[1])


def test_parallel_linear_probe_config_round_trips() -> None:
    config = ParallelLinearProbeConfig(
        input_size=12,
        hidden_size=16,
        dropout=0.1,
        num_labels=4,
        n_layers=2,
        task_type='multilabel',
        pre_ln=False,
        use_bias=True,
        num_runs=3,
        run_seeds=[5, 6, 7],
    )

    serialized = config.to_dict()
    rebuilt = ParallelLinearProbeConfig(**serialized)

    assert rebuilt.input_size == 12
    assert rebuilt.hidden_size == 16
    assert rebuilt.dropout == pytest.approx(0.1)
    assert rebuilt.num_labels == 4
    assert rebuilt.n_layers == 2
    assert rebuilt.task_type == 'multilabel'
    assert not rebuilt.pre_ln
    assert rebuilt.use_bias
    assert rebuilt.num_runs == 3
    assert rebuilt.run_seeds == [5, 6, 7]


def test_parallel_linear_probe_package_exports() -> None:
    assert "ParallelLinearProbe" in probes_package.__all__
    assert "ParallelLinearProbeConfig" in probes_package.__all__
    assert "ParallelLinearProbeEnsemble" in probes_package.__all__
    assert probes_package.ParallelLinearProbe is ParallelLinearProbe
    assert probes_package.ParallelLinearProbeConfig is ParallelLinearProbeConfig
    assert probes_package.ParallelLinearProbeEnsemble is ParallelLinearProbeEnsemble


def test_parallel_linear_probe_exports_matching_single_run_probe() -> None:
    config = ParallelLinearProbeConfig(
        input_size=6,
        hidden_size=10,
        dropout=0.0,
        num_labels=1,
        n_layers=1,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[101, 102, 103],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    single_probe = parallel_probe.to_mlp_probe(1).eval()
    embeddings = torch.randn(9, 6)

    parallel_logits = parallel_probe(embeddings=embeddings).logits[:, 1, :]
    single_logits = single_probe(embeddings=embeddings).logits

    assert torch.allclose(parallel_logits, single_logits, atol=1e-6)


@pytest.mark.parametrize(
    "task_type,num_labels",
    [
        ('singlelabel', 3),
        ('multilabel', 3),
        ('regression', 1),
        ('sigmoid_regression', 1),
    ],
)
def test_parallel_linear_probe_ensemble_matches_average_exported_single_probes(task_type, num_labels) -> None:
    config = ParallelLinearProbeConfig(
        input_size=6,
        hidden_size=10,
        dropout=0.0,
        num_labels=num_labels,
        n_layers=1,
        task_type=task_type,
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[1011, 1012, 1013],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    ensemble = parallel_probe.to_ensemble().eval()
    embeddings = torch.randn(7, 6)

    ensemble_logits = ensemble(embeddings=embeddings).logits
    single_logits = []
    for run_idx in range(config.num_runs):
        single_probe = parallel_probe.to_mlp_probe(run_idx).eval()
        single_logits.append(single_probe(embeddings=embeddings).logits)
    expected = torch.stack(single_logits, dim=1).mean(dim=1)

    assert torch.allclose(ensemble_logits, expected, atol=1e-6)


def test_parallel_linear_probe_ensemble_can_average_selected_runs() -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=4,
        run_seeds=[2011, 2012, 2013, 2014],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    ensemble = parallel_probe.to_ensemble(run_indices=[1, 3]).eval()
    embeddings = torch.randn(6, 5)

    output = ensemble(embeddings=embeddings)
    expected = torch.stack(
        [
            parallel_probe.to_mlp_probe(1).eval()(embeddings=embeddings).logits,
            parallel_probe.to_mlp_probe(3).eval()(embeddings=embeddings).logits,
        ],
        dim=1,
    ).mean(dim=1)

    assert ensemble.run_indices == (1, 3)
    assert torch.allclose(output.logits, expected, atol=1e-6)


def test_parallel_linear_probe_ensemble_accepts_run_specific_embeddings() -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[3011, 3012, 3013],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    ensemble = parallel_probe.to_ensemble().eval()
    embeddings = torch.randn(4, 3, 5)

    output = ensemble(embeddings=embeddings)
    expected = torch.stack(
        [
            parallel_probe.to_mlp_probe(run_idx).eval()(embeddings=embeddings[:, run_idx, :]).logits
            for run_idx in range(config.num_runs)
        ],
        dim=1,
    ).mean(dim=1)

    assert torch.allclose(output.logits, expected, atol=1e-6)


def test_parallel_linear_probe_ensemble_probability_mode_averages_probabilities() -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=8,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[4011, 4012, 4013],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    ensemble = parallel_probe.to_ensemble(average_mode='probabilities').eval()
    embeddings = torch.randn(5, 5)

    output = ensemble(embeddings=embeddings)
    expected = torch.softmax(parallel_probe(embeddings=embeddings).logits, dim=-1).mean(dim=1)

    assert torch.allclose(output.logits, expected, atol=1e-6)


def test_parallel_linear_probe_ensemble_reports_loss_for_logit_average() -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=8,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[5011, 5012, 5013],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    ensemble = parallel_probe.to_ensemble().eval()
    embeddings = torch.randn(5, 5)
    labels = torch.tensor([0, 1, 2, 0, 1])

    output = ensemble(embeddings=embeddings, labels=labels)
    expected = F.cross_entropy(output.logits, labels)

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_ensemble_rejects_invalid_configuration() -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=8,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[6011, 6012],
    )
    parallel_probe = ParallelLinearProbe(config).eval()

    with pytest.raises(AssertionError, match="at least one"):
        parallel_probe.to_ensemble(run_indices=[])
    with pytest.raises(AssertionError, match="run index"):
        parallel_probe.to_ensemble(run_indices=[2])
    with pytest.raises(AssertionError, match="average_mode"):
        parallel_probe.to_ensemble(average_mode='votes')
    with pytest.raises(AssertionError, match="Singlelabel ensemble loss"):
        parallel_probe.to_ensemble(average_mode='probabilities')(
            embeddings=torch.randn(3, 5),
            labels=torch.tensor([0, 1, 2]),
        )


def test_parallel_linear_probe_seeded_run_matches_linear_probe_initialization() -> None:
    seed = 77
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=7,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[seed, seed + 1],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        single_probe = MLPProbe(
            MLPProbeConfig(
                input_size=5,
                hidden_size=7,
                dropout=0.0,
                num_labels=2,
                n_layers=0,
                task_type='singlelabel',
                pre_ln=True,
                use_bias=True,
            )
        ).eval()
    embeddings = torch.randn(11, 5)

    parallel_logits = parallel_probe(embeddings=embeddings).logits[:, 0, :]
    single_logits = single_probe(embeddings=embeddings).logits

    assert torch.allclose(parallel_logits, single_logits, atol=1e-6)


def _parallel_probe_task_labels(task_type: str, num_labels: int, num_runs: int = 2):
    if task_type == 'singlelabel':
        shared_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        run_labels = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [0, 2],
            ],
            dtype=torch.long,
        )
    elif task_type == 'multilabel':
        shared_labels = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        run_labels = torch.stack([shared_labels, 1.0 - shared_labels], dim=1)
    elif task_type == 'regression':
        shared_labels = torch.tensor([0.1, -0.2, 0.5, 1.0])
        run_labels = torch.stack([shared_labels, shared_labels + 0.25], dim=1)
    else:
        assert task_type == 'sigmoid_regression'
        shared_labels = torch.tensor([0.1, 0.2, 0.7, 0.9])
        run_labels = torch.stack([shared_labels, 1.0 - shared_labels], dim=1)

    assert run_labels.shape[1] == num_runs
    if task_type in ('singlelabel', 'multilabel'):
        assert num_labels == 3
    else:
        assert num_labels == 1
    return shared_labels, run_labels


@pytest.mark.parametrize(
    ("task_type", "num_labels"),
    [
        ('singlelabel', 3),
        ('multilabel', 3),
        ('regression', 1),
        ('sigmoid_regression', 1),
    ],
)
def test_parallel_linear_probe_shared_eval_matches_exported_single_probes(task_type, num_labels) -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=7,
        dropout=0.0,
        num_labels=num_labels,
        n_layers=0,
        task_type=task_type,
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[301, 302],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(4, 5)
    shared_labels, _run_labels = _parallel_probe_task_labels(task_type, num_labels)

    parallel_output = parallel_probe(embeddings=embeddings, labels=shared_labels)
    single_losses = []
    for run_idx in range(config.num_runs):
        single_probe = parallel_probe.to_mlp_probe(run_idx).eval()
        single_output = single_probe(embeddings=embeddings, labels=shared_labels)
        assert torch.allclose(parallel_output.logits[:, run_idx, :], single_output.logits, atol=1e-6)
        single_losses.append(single_output.loss)

    expected_loss = torch.stack(single_losses).mean()
    assert torch.allclose(parallel_output.loss, expected_loss, atol=1e-6)


@pytest.mark.parametrize(
    ("task_type", "num_labels"),
    [
        ('singlelabel', 3),
        ('multilabel', 3),
        ('regression', 1),
        ('sigmoid_regression', 1),
    ],
)
def test_parallel_linear_probe_run_specific_eval_matches_exported_single_probes(task_type, num_labels) -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=7,
        dropout=0.0,
        num_labels=num_labels,
        n_layers=0,
        task_type=task_type,
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[401, 402],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(4, 2, 5)
    _shared_labels, run_labels = _parallel_probe_task_labels(task_type, num_labels)

    parallel_output = parallel_probe(embeddings=embeddings, labels=run_labels)
    single_losses = []
    for run_idx in range(config.num_runs):
        single_probe = parallel_probe.to_mlp_probe(run_idx).eval()
        if task_type == 'singlelabel':
            single_labels = run_labels[:, run_idx]
        elif task_type == 'multilabel':
            single_labels = run_labels[:, run_idx, :]
        else:
            single_labels = run_labels[:, run_idx]
        single_output = single_probe(embeddings=embeddings[:, run_idx, :], labels=single_labels)
        assert torch.allclose(parallel_output.logits[:, run_idx, :], single_output.logits, atol=1e-6)
        single_losses.append(single_output.loss)

    expected_loss = torch.stack(single_losses).mean()
    assert torch.allclose(parallel_output.loss, expected_loss, atol=1e-6)


def test_parallel_linear_probe_accepts_run_specific_batches() -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=7,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[501, 502],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(6, 2, 5)
    labels = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 2],
            [1, 0],
            [2, 1],
        ]
    )

    output = parallel_probe(embeddings=embeddings, labels=labels)

    assert output.logits.shape == (6, 2, 3)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    for run_idx in range(config.num_runs):
        single_probe = parallel_probe.to_mlp_probe(run_idx).eval()
        single_logits = single_probe(embeddings=embeddings[:, run_idx, :]).logits
        assert torch.allclose(output.logits[:, run_idx, :], single_logits, atol=1e-6)


def test_parallel_linear_probe_run_specific_training_loss_is_sum_by_run() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=6,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[601, 602, 603],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(5, 3, 4)
    labels = torch.randn(5, 3)

    output = model(embeddings=embeddings, labels=labels)
    expected = torch.zeros((), dtype=output.loss.dtype)
    for run_idx in range(config.num_runs):
        expected = expected + F.mse_loss(output.logits[:, run_idx, :], labels[:, run_idx].unsqueeze(-1))

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_multilabel_shared_labels_when_num_labels_equals_num_runs() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=6,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='multilabel',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[701, 702, 703],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(5, 4)
    labels = torch.randint(0, 2, (5, 3)).float()

    output = model(embeddings=embeddings, labels=labels)
    expected = torch.zeros((), dtype=output.loss.dtype)
    for run_idx in range(config.num_runs):
        expected = expected + F.binary_cross_entropy_with_logits(output.logits[:, run_idx, :], labels)

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_regression_shared_labels_when_num_labels_equals_num_runs() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=6,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[801, 802],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(5, 4)
    labels = torch.randn(5, 2)

    output = model(embeddings=embeddings, labels=labels)
    expected = torch.zeros((), dtype=output.loss.dtype)
    for run_idx in range(config.num_runs):
        expected = expected + F.mse_loss(output.logits[:, run_idx, :], labels)

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_multilabel_run_specific_labels_are_explicitly_3d() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=6,
        dropout=0.0,
        num_labels=3,
        n_layers=0,
        task_type='multilabel',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[901, 902],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(5, 2, 4)
    labels = torch.randint(0, 2, (5, 2, 3)).float()

    output = model(embeddings=embeddings, labels=labels)
    expected = torch.zeros((), dtype=output.loss.dtype)
    for run_idx in range(config.num_runs):
        expected = expected + F.binary_cross_entropy_with_logits(output.logits[:, run_idx, :], labels[:, run_idx, :])

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_rejects_ambiguous_multilabel_run_labels() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=6,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='multilabel',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[1001, 1002, 1003],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(5, 3, 4)
    labels = torch.randint(0, 2, (5, 3)).float()

    with pytest.raises(AssertionError, match="shared multilabel labels"):
        model(embeddings=embeddings, labels=labels)


def test_parallel_linear_probe_save_pretrained_round_trip(tmp_path) -> None:
    config = ParallelLinearProbeConfig(
        input_size=5,
        hidden_size=9,
        dropout=0.0,
        num_labels=2,
        n_layers=1,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[901, 902, 903],
    )
    model = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(4, 5)
    before = model(embeddings=embeddings).logits

    model.save_pretrained(tmp_path)
    reloaded = ParallelLinearProbe.from_pretrained(tmp_path).eval()
    after = reloaded(embeddings=embeddings).logits

    assert reloaded.config.num_runs == 3
    assert reloaded.config.run_seeds == [901, 902, 903]
    assert torch.allclose(before, after, atol=1e-6)


def test_parallel_linear_probe_loss_updates_each_run_bank() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[201, 202, 203],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(6, 4)
    labels = torch.randn(6)

    loss = model(embeddings=embeddings, labels=labels).loss
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_parallel_probe_per_run_gradient_clipping_clips_each_bank_independently() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=6,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[1001, 1002],
    )
    model = ParallelLinearProbe(config)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)

    first_layer = model._parameter_layers()[0]
    first_layer.weight.grad[0].fill_(3.0)
    first_layer.weight.grad[1].fill_(0.1)
    first_layer.bias.grad[0].fill_(4.0)
    first_layer.bias.grad[1].fill_(0.1)

    pre_clip_norms = _clip_parallel_probe_gradients_per_run(model, max_norm=1.0)

    assert pre_clip_norms[0] > 1.0
    assert pre_clip_norms[1] < 1.0
    post_clip_norms = []
    for run_idx in range(model.num_runs):
        norm_sq = torch.zeros(())
        for layer in model._parameter_layers():
            if layer.weight.grad is not None:
                norm_sq = norm_sq + layer.weight.grad[run_idx].pow(2).sum()
            if layer.bias is not None and layer.bias.grad is not None:
                norm_sq = norm_sq + layer.bias.grad[run_idx].pow(2).sum()
        post_clip_norms.append(float(norm_sq.sqrt().item()))

    assert post_clip_norms[0] == pytest.approx(1.0, abs=1e-5)
    assert post_clip_norms[1] == pytest.approx(pre_clip_norms[1], abs=1e-6)


def test_parallel_linear_probe_training_loss_is_sum_of_per_run_losses() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[301, 302, 303],
    )
    model = ParallelLinearProbe(config)
    embeddings = torch.randn(5, 4)
    labels = torch.randn(5)

    output = model(embeddings=embeddings, labels=labels)
    expected = torch.zeros((), dtype=output.loss.dtype)
    target = labels.unsqueeze(-1)
    for run_idx in range(config.num_runs):
        expected = expected + F.mse_loss(output.logits[:, run_idx, :], target)

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_eval_loss_is_mean_of_per_run_losses() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='regression',
        pre_ln=True,
        use_bias=True,
        num_runs=3,
        run_seeds=[401, 402, 403],
    )
    model = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(5, 4)
    labels = torch.randn(5)

    output = model(embeddings=embeddings, labels=labels)
    expected = torch.zeros((), dtype=output.loss.dtype)
    target = labels.unsqueeze(-1)
    for run_idx in range(config.num_runs):
        expected = expected + F.mse_loss(output.logits[:, run_idx, :], target)
    expected = expected / config.num_runs

    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_parallel_linear_probe_sigmoid_regression_ignores_masked_shared_labels() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='sigmoid_regression',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[501, 502],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(4, 4)
    labels = torch.tensor([0.1, -100.0, 0.8, 0.4])

    output = parallel_probe(embeddings=embeddings, labels=labels)
    single_losses = []
    for run_idx in range(config.num_runs):
        single_probe = parallel_probe.to_mlp_probe(run_idx).eval()
        single_output = single_probe(embeddings=embeddings, labels=labels)
        single_losses.append(single_output.loss)

    assert torch.allclose(output.loss, torch.stack(single_losses).mean(), atol=1e-6)


def test_parallel_linear_probe_sigmoid_regression_ignores_masked_run_specific_labels() -> None:
    config = ParallelLinearProbeConfig(
        input_size=4,
        hidden_size=8,
        dropout=0.0,
        num_labels=1,
        n_layers=0,
        task_type='sigmoid_regression',
        pre_ln=True,
        use_bias=True,
        num_runs=2,
        run_seeds=[503, 504],
    )
    parallel_probe = ParallelLinearProbe(config).eval()
    embeddings = torch.randn(4, 2, 4)
    labels = torch.tensor(
        [
            [0.1, -100.0],
            [0.2, -100.0],
            [0.7, -100.0],
            [0.8, -100.0],
        ]
    )

    output = parallel_probe(embeddings=embeddings, labels=labels)
    single_losses = []
    for run_idx in range(config.num_runs):
        single_probe = parallel_probe.to_mlp_probe(run_idx).eval()
        single_output = single_probe(
            embeddings=embeddings[:, run_idx, :],
            labels=labels[:, run_idx],
        )
        single_losses.append(single_output.loss)

    assert torch.allclose(output.loss, torch.stack(single_losses).mean(), atol=1e-6)


def test_parallel_probe_trainer_eligibility_gate() -> None:
    mixin = _trainer_mixin_for_parallel_tests()

    assert mixin._can_parallelize_probe_runs(full=False)
    assert not mixin._can_parallelize_probe_runs(full=False, ppi=True)

    mixin.probe_args.tokenwise = True
    assert not mixin._can_parallelize_probe_runs(full=False)
    mixin.probe_args.tokenwise = False

    mixin.probe_args.probe_type = 'transformer'
    assert not mixin._can_parallelize_probe_runs(full=False)
    mixin.probe_args.probe_type = 'mlp'

    assert not mixin._can_parallelize_probe_runs(full=True)

    mixin.embedding_args.matrix_embed = True
    assert not mixin._can_parallelize_probe_runs(full=False)
    mixin.embedding_args.matrix_embed = False

    mixin.trainer_args.save = True
    assert mixin._can_parallelize_probe_runs(full=False)
    mixin.trainer_args.save = False

    mixin.trainer_args.num_runs = 1
    assert not mixin._can_parallelize_probe_runs(full=False)


def test_parallel_trainer_restores_eval_output_multiplier_on_error(monkeypatch) -> None:
    class FailingTrainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("forced before training")

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.eval_output_multiplier = 7
    monkeypatch.setattr(trainers_module, "Trainer", FailingTrainer)

    with pytest.raises(RuntimeError, match="forced before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=[(torch.zeros(4), torch.tensor(0))],
            valid_dataset=[(torch.zeros(4), torch.tensor(0))],
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    assert mixin.trainer_args.eval_output_multiplier == 7


def test_parallel_trainer_uses_shared_train_dataset_by_default(monkeypatch) -> None:
    captured = {}

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            captured['train_dataset'] = kwargs['train_dataset']
            raise RuntimeError("stop before training")

    mixin = _trainer_mixin_for_parallel_tests()
    train_dataset = [(torch.zeros(4), torch.tensor(0)), (torch.ones(4), torch.tensor(1))]
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    with pytest.raises(RuntimeError, match="stop before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=train_dataset,
            valid_dataset=[(torch.zeros(4), torch.tensor(0))],
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    assert captured['train_dataset'] is train_dataset


def test_parallel_trainer_passes_parallel_max_grad_norm(monkeypatch) -> None:
    captured = {}

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            captured['max_grad_norm'] = kwargs['args'].max_grad_norm
            raise RuntimeError("stop before training")

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.parallel_probe_max_grad_norm = 0.5
    train_dataset = [(torch.zeros(4), torch.tensor(0)), (torch.ones(4), torch.tensor(1))]
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    with pytest.raises(RuntimeError, match="stop before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=train_dataset,
            valid_dataset=[(torch.zeros(4), torch.tensor(0))],
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    assert captured['max_grad_norm'] == pytest.approx(0.5)


def test_parallel_trainer_per_run_grad_clip_installs_callback(monkeypatch) -> None:
    captured = {}

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            captured['max_grad_norm'] = kwargs['args'].max_grad_norm
            captured['callbacks'] = kwargs['callbacks']
            raise RuntimeError("stop before training")

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.parallel_probe_max_grad_norm = 0.5
    mixin.trainer_args.parallel_probe_grad_clip_mode = 'per_run'
    train_dataset = [(torch.zeros(4), torch.tensor(0)), (torch.ones(4), torch.tensor(1))]
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    with pytest.raises(RuntimeError, match="stop before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=train_dataset,
            valid_dataset=[(torch.zeros(4), torch.tensor(0))],
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    assert captured['max_grad_norm'] == pytest.approx(0.0)
    assert any(isinstance(callback, ParallelProbePerRunGradientClipCallback) for callback in captured['callbacks'])


def test_parallel_trainer_passes_per_run_compute_metrics(monkeypatch) -> None:
    captured = {}

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            captured['compute_metrics'] = kwargs['compute_metrics']
            raise RuntimeError("stop before training")

    mixin = _trainer_mixin_for_parallel_tests()
    train_dataset = [(torch.zeros(4), torch.tensor(0)), (torch.ones(4), torch.tensor(1))]
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    with pytest.raises(RuntimeError, match="stop before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=train_dataset,
            valid_dataset=[(torch.zeros(4), torch.tensor(0))],
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    labels = np.array([0, 1, 0, 1])
    logits = np.array(
        [
            [[3.0, 0.0], [0.0, 3.0]],
            [[0.0, 3.0], [3.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    metrics = captured['compute_metrics'](
        SimpleNamespace(predictions=logits, label_ids=labels)
    )

    assert metrics['run_0_accuracy'] == 1.0
    assert metrics['run_1_accuracy'] == 0.0
    assert metrics['run_0_loss'] < metrics['run_1_loss']


def test_parallel_trainer_uses_validation_for_initial_eval_and_clears_predict_metrics(monkeypatch) -> None:
    captured = {'predict_compute_metrics': []}

    class FakeAccelerator:
        def free_memory(self):
            pass

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs['model']
            self.compute_metrics = kwargs['compute_metrics']
            self.accelerator = FakeAccelerator()

        def evaluate(self, dataset):
            captured['initial_eval_dataset'] = dataset
            captured['initial_eval_has_compute_metrics'] = self.compute_metrics is not None
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': 1.0})

        def predict(self, dataset):
            captured['predict_compute_metrics'].append(self.compute_metrics)
            logits = np.array(
                [
                    [[3.0, 0.0], [0.0, 3.0]],
                    [[0.0, 3.0], [3.0, 0.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                    [[0.0, 2.0], [2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            return SimpleNamespace(predictions=logits, label_ids=labels)

    mixin = _trainer_mixin_for_parallel_tests()
    train_dataset = [(torch.zeros(4), torch.tensor(0)), (torch.ones(4), torch.tensor(1))]
    valid_dataset = [
        (torch.zeros(4), torch.tensor(0)),
        (torch.ones(4), torch.tensor(1)),
        (torch.full((4,), 2.0), torch.tensor(0)),
        (torch.full((4,), 3.0), torch.tensor(1)),
    ]
    test_dataset = list(valid_dataset)
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    mixin._train_parallel_linear_probe_runs(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )

    assert captured['initial_eval_dataset'] is valid_dataset
    assert captured['initial_eval_has_compute_metrics'] is True
    assert captured['predict_compute_metrics'] == [None, None]


def test_parallel_probe_compute_metrics_uses_requested_split_name() -> None:
    captured = {}
    mixin = _trainer_mixin_for_parallel_tests()

    def fake_metrics_by_run(logits, labels, data_name, split_name):
        captured['logits_shape'] = logits.shape
        captured['labels_shape'] = labels.shape
        captured['data_name'] = data_name
        captured['split_name'] = split_name
        return [
            {
                'eval_loss': np.float32(1.25),
                'eval_accuracy': np.float64(0.75),
            },
        ]

    mixin._parallel_probe_metrics_by_run = fake_metrics_by_run
    compute_metrics = mixin._parallel_probe_compute_metrics('data', 'eval')
    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.zeros((3, 1, 2), dtype=np.float32),
            label_ids=np.array([0, 1, 0], dtype=np.int64),
        )
    )

    assert captured['logits_shape'] == (3, 1, 2)
    assert captured['labels_shape'] == (3,)
    assert captured['data_name'] == 'data'
    assert captured['split_name'] == 'eval'
    assert metrics == {
        'run_0_loss': 1.25,
        'run_0_accuracy': 0.75,
    }


def test_parallel_trainer_sigmoid_loss_reporting_ignores_masked_labels() -> None:
    mixin = _trainer_mixin_for_parallel_tests(task_type='sigmoid_regression')
    logits = np.array(
        [
            [[0.1], [0.8]],
            [[0.2], [0.7]],
            [[0.9], [0.6]],
            [[0.8], [0.5]],
        ],
        dtype=np.float32,
    )
    labels = np.array([0.0, -100.0, 1.0, -100.0], dtype=np.float32)

    losses = mixin._parallel_probe_losses_by_run(logits, labels, 'sigmoid_regression')

    expected_run_0 = F.binary_cross_entropy(
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.0, 1.0]),
    )
    expected_run_1 = F.binary_cross_entropy(
        torch.tensor([0.8, 0.6]),
        torch.tensor([0.0, 1.0]),
    )
    assert losses[0] == pytest.approx(float(expected_run_0.item()))
    assert losses[1] == pytest.approx(float(expected_run_1.item()))


def test_parallel_probe_json_safe_metadata_converts_numpy_values() -> None:
    mixin = _trainer_mixin_for_parallel_tests()

    safe = mixin._parallel_probe_json_safe(
        {
            "scalar": np.float32(1.25),
            "array": np.array([np.int64(1), np.int64(2)]),
            "tuple": (np.float64(3.5),),
        }
    )

    json.dumps(safe)
    assert safe["scalar"] == pytest.approx(1.25)
    assert safe["array"] == [1, 2]
    assert safe["tuple"][0] == pytest.approx(3.5)


def test_parallel_trainer_wraps_train_dataset_for_run_specific_mode(monkeypatch) -> None:
    captured = {}

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            captured['train_dataset'] = kwargs['train_dataset']
            captured['eval_dataset'] = kwargs['eval_dataset']
            raise RuntimeError("stop before training")

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.parallel_probe_batch_mode = 'run_specific'
    train_dataset = [
        (torch.zeros(4), torch.tensor(0)),
        (torch.ones(4), torch.tensor(1)),
        (torch.full((4,), 2.0), torch.tensor(0)),
    ]
    valid_dataset = [(torch.zeros(4), torch.tensor(0))]
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    with pytest.raises(RuntimeError, match="stop before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    assert isinstance(captured['train_dataset'], ParallelRunDataset)
    assert captured['train_dataset'].run_seeds == (42, 43)
    assert captured['train_dataset'].base_dataset is train_dataset
    assert captured['eval_dataset'] is valid_dataset


def test_parallel_trainer_passes_affine_index_strategy(monkeypatch) -> None:
    captured = {}

    class InspectingTrainer:
        def __init__(self, *args, **kwargs):
            captured['train_dataset'] = kwargs['train_dataset']
            raise RuntimeError("stop before training")

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.parallel_probe_batch_mode = 'run_specific'
    mixin.trainer_args.parallel_probe_index_strategy = 'affine'
    train_dataset = [
        (torch.zeros(4), torch.tensor(0)),
        (torch.ones(4), torch.tensor(1)),
        (torch.full((4,), 2.0), torch.tensor(0)),
    ]
    monkeypatch.setattr(trainers_module, "Trainer", InspectingTrainer)

    with pytest.raises(RuntimeError, match="stop before training"):
        mixin._train_parallel_linear_probe_runs(
            train_dataset=train_dataset,
            valid_dataset=[(torch.zeros(4), torch.tensor(0))],
            test_dataset=[(torch.zeros(4), torch.tensor(0))],
            data_collator=None,
            tokenizer=None,
            log_id='log',
            model_name='model',
            data_name='data',
        )

    assert isinstance(captured['train_dataset'], ParallelRunDataset)
    assert captured['train_dataset'].index_strategy == 'affine'
    assert captured['train_dataset'].index_memory_bytes == 0


def test_parallel_trainer_reports_parallel_execution_metadata(monkeypatch) -> None:
    captured = {}

    class FakeAccelerator:
        def free_memory(self):
            captured['freed_memory'] = True

    class CompiledModelWrapper:
        def __init__(self, original_model):
            self._orig_mod = original_model

    class MetadataTrainer:
        def __init__(self, *args, **kwargs):
            self.model = CompiledModelWrapper(kwargs['model'])
            captured['max_grad_norm'] = kwargs['args'].max_grad_norm
            self.accelerator = FakeAccelerator()

        def evaluate(self, dataset):
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': 8.0})

        def predict(self, dataset):
            logits = np.array(
                [
                    [[3.0, 0.0], [0.0, 3.0]],
                    [[0.0, 3.0], [3.0, 0.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                    [[0.0, 2.0], [2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            return SimpleNamespace(predictions=logits, label_ids=labels)

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.parallel_probe_batch_mode = 'run_specific'
    mixin.trainer_args.parallel_probe_index_strategy = 'affine'
    monkeypatch.setattr(trainers_module, "Trainer", MetadataTrainer)

    model, _valid_metrics, test_metrics, _y_pred, _y_true = mixin._train_parallel_linear_probe_runs(
        train_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
        ],
        valid_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        test_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )

    assert isinstance(model, MLPProbe)
    assert test_metrics['training_time_seconds'] == pytest.approx(8.0)
    assert test_metrics['parallel_probe_num_runs'] == 2
    assert test_metrics['parallel_probe_seconds_per_run'] == pytest.approx(4.0)
    assert test_metrics['parallel_probe_batch_mode'] == 'run_specific'
    assert test_metrics['parallel_probe_index_strategy'] == 'affine'
    assert test_metrics['parallel_probe_estimated_peak_budget_gb'] is None
    assert test_metrics['parallel_probe_max_grad_norm'] == pytest.approx(0.0)
    assert test_metrics['parallel_probe_grad_clip_mode'] == 'global'
    assert captured['max_grad_norm'] == pytest.approx(0.0)
    assert test_metrics['parallel_probe_estimated_parameter_count'] > 0
    assert test_metrics['parallel_probe_estimated_training_state_bytes'] > 0
    assert test_metrics['parallel_probe_estimated_peak_group_training_state_bytes'] > 0
    assert test_metrics['parallel_probe_estimated_batch_activation_bytes'] > 0
    assert test_metrics['parallel_probe_estimated_peak_group_batch_activation_bytes'] > 0
    assert test_metrics['parallel_probe_estimated_run_specific_index_bytes'] == 0
    assert test_metrics['parallel_probe_estimated_peak_group_bytes'] > (
        test_metrics['parallel_probe_estimated_peak_group_training_state_bytes']
    )
    assert test_metrics['parallel_probe_estimated_forward_flops_per_batch'] > 0
    assert test_metrics['parallel_probe_estimated_peak_group_forward_flops_per_batch'] > 0
    assert test_metrics['parallel_probe_estimated_training_flops_per_batch'] == (
        test_metrics['parallel_probe_estimated_forward_flops_per_batch'] * 3
    )
    assert test_metrics['parallel_probe_estimated_peak_group_training_flops_per_batch'] == (
        test_metrics['parallel_probe_estimated_peak_group_forward_flops_per_batch'] * 3
    )
    assert test_metrics['parallel_probe_index_memory_bytes'] == 0
    assert test_metrics['parallel_probe_total_runs'] == 2
    assert test_metrics['parallel_probe_vectorized_runs'] == 2
    assert test_metrics['parallel_probe_sequential_runs'] == 0
    assert test_metrics['parallel_probe_trainer_invocations'] == 1
    assert test_metrics['parallel_probe_invocation_reduction'] == 1
    assert test_metrics['parallel_probe_compression_ratio'] == pytest.approx(2.0)
    assert test_metrics['parallel_probe_run_seeds'] == [42, 43]
    assert test_metrics['parallel_probe_best_selection_metric'] == 'test_loss'
    records = test_metrics['parallel_probe_run_records']
    assert len(records) == 2
    assert records[0]['run_index'] == 0
    assert records[0]['run_number'] == 1
    assert records[0]['seed'] == 42
    assert records[0]['run_id'] == 'data/model/seed-42'
    assert records[0]['group_index'] == 0
    assert records[0]['group_number'] == 1
    assert records[0]['local_run_index'] == 0
    assert records[0]['local_run_number'] == 1
    assert records[0]['valid_loss'] == pytest.approx(records[0]['test_loss'])
    assert records[1]['run_index'] == 1
    assert records[1]['run_number'] == 2
    assert records[1]['seed'] == 43
    assert records[1]['run_id'] == 'data/model/seed-43'
    assert records[1]['group_index'] == 0
    assert records[1]['group_number'] == 1
    assert records[1]['local_run_index'] == 1
    assert records[1]['local_run_number'] == 2
    assert records[1]['valid_loss'] == pytest.approx(records[1]['test_loss'])
    assert records[0]['test_loss'] < records[1]['test_loss']
    assert len(test_metrics['parallel_probe_valid_run_metrics']) == 2
    assert len(test_metrics['parallel_probe_test_run_metrics']) == 2
    assert test_metrics['parallel_probe_valid_run_metrics'][0]['eval_loss'] == pytest.approx(records[0]['valid_loss'])
    assert test_metrics['parallel_probe_test_run_metrics'][0]['test_loss'] == pytest.approx(records[0]['test_loss'])
    assert test_metrics['parallel_probe_best_run_index'] == 0
    assert test_metrics['parallel_probe_best_run_number'] == 1
    assert test_metrics['parallel_probe_best_run_id'] == 'data/model/seed-42'
    assert test_metrics['parallel_probe_best_seed'] == 42
    assert test_metrics['parallel_probe_best_test_loss'] < 0.2
    assert captured['freed_memory']


def test_parallel_trainer_chunks_seed_groups_and_aggregates_metadata(monkeypatch) -> None:
    captured = {'freed_memory_count': 0, 'groups': [], 'output_dirs': []}

    class FakeAccelerator:
        def free_memory(self):
            captured['freed_memory_count'] += 1

    class ChunkedTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs['model']
            self.run_seeds = list(self.model.config.run_seeds)
            captured['groups'].append(self.run_seeds)
            captured['output_dirs'].append(kwargs['args'].output_dir)
            self.accelerator = FakeAccelerator()

        def evaluate(self, dataset):
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': float(len(self.run_seeds))})

        def predict(self, dataset):
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            logits = np.zeros((4, len(self.run_seeds), 2), dtype=np.float32)
            correct_logits = np.array(
                [
                    [5.0, 0.0],
                    [0.0, 5.0],
                    [5.0, 0.0],
                    [0.0, 5.0],
                ],
                dtype=np.float32,
            )
            incorrect_logits = np.array(
                [
                    [0.0, 5.0],
                    [5.0, 0.0],
                    [0.0, 5.0],
                    [5.0, 0.0],
                ],
                dtype=np.float32,
            )
            for local_idx, seed in enumerate(self.run_seeds):
                if seed == 44:
                    logits[:, local_idx, :] = correct_logits
                else:
                    logits[:, local_idx, :] = incorrect_logits
            return SimpleNamespace(predictions=logits, label_ids=labels)

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.num_runs = 5
    mixin.trainer_args.parallel_probe_max_group_size = 2
    monkeypatch.setattr(trainers_module, "Trainer", ChunkedTrainer)

    model, _valid_metrics, test_metrics, y_pred, y_true = mixin._train_parallel_linear_probe_runs(
        train_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
        ],
        valid_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        test_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )

    assert isinstance(model, MLPProbe)
    assert captured['groups'] == [[42, 43], [44, 45], [46]]
    assert captured['output_dirs'][0].endswith('parallel_group_1')
    assert captured['output_dirs'][1].endswith('parallel_group_2')
    assert captured['output_dirs'][2].endswith('parallel_group_3')
    assert captured['freed_memory_count'] == 3
    assert test_metrics['training_time_seconds'] == pytest.approx(5.0)
    assert test_metrics['parallel_probe_num_runs'] == 5
    assert test_metrics['parallel_probe_seconds_per_run'] == pytest.approx(1.0)
    assert test_metrics['parallel_probe_max_group_size'] == 2
    assert test_metrics['parallel_probe_explicit_max_group_size'] == 2
    assert test_metrics['parallel_probe_training_state_budget_group_size'] is None
    assert test_metrics['parallel_probe_estimated_peak_budget_group_size'] is None
    assert test_metrics['parallel_probe_group_size_candidates'] == [2]
    assert test_metrics['parallel_probe_peak_budget_includes_index'] is False
    assert test_metrics['parallel_probe_group_sizes'] == [2, 2, 1]
    assert test_metrics['parallel_probe_group_run_seeds'] == [[42, 43], [44, 45], [46]]
    assert test_metrics['parallel_probe_group_output_dirs'] == captured['output_dirs']
    assert len(test_metrics['parallel_probe_group_runtime_records']) == 3
    assert test_metrics['parallel_probe_group_runtime_records'][0]['group_number'] == 1
    assert test_metrics['parallel_probe_group_runtime_records'][0]['execution_kind'] == 'vectorized'
    assert test_metrics['parallel_probe_group_runtime_records'][0]['num_runs'] == 2
    assert test_metrics['parallel_probe_group_runtime_records'][0]['run_seeds'] == [42, 43]
    assert test_metrics['parallel_probe_group_runtime_records'][0]['train_runtime_seconds'] == pytest.approx(2.0)
    assert test_metrics['parallel_probe_group_runtime_records'][0]['seconds_per_run'] == pytest.approx(1.0)
    assert test_metrics['parallel_probe_group_runtime_records'][0]['output_dir'] == captured['output_dirs'][0]
    assert test_metrics['parallel_probe_group_runtime_records'][0]['index_memory_bytes'] == 0
    assert test_metrics['parallel_probe_group_runtime_records'][0]['estimated_peak_bytes'] > 0
    assert test_metrics['parallel_probe_group_runtime_records'][2]['execution_kind'] == 'eligible_singleton'
    assert test_metrics['parallel_probe_group_runtime_records'][2]['num_runs'] == 1
    assert test_metrics['parallel_probe_group_runtime_records'][2]['seconds_per_run'] == pytest.approx(1.0)
    assert test_metrics['parallel_probe_total_runs'] == 5
    assert test_metrics['parallel_probe_vectorized_runs'] == 4
    assert test_metrics['parallel_probe_sequential_runs'] == 1
    assert test_metrics['parallel_probe_trainer_invocations'] == 3
    assert test_metrics['parallel_probe_invocation_reduction'] == 2
    assert test_metrics['parallel_probe_compression_ratio'] == pytest.approx(5.0 / 3.0)
    assert test_metrics['parallel_probe_run_seeds'] == [42, 43, 44, 45, 46]
    assert len(test_metrics['parallel_probe_run_records']) == 5
    assert test_metrics['parallel_probe_run_records'][0]['seed'] == 42
    assert test_metrics['parallel_probe_run_records'][0]['group_number'] == 1
    assert test_metrics['parallel_probe_run_records'][0]['local_run_number'] == 1
    assert test_metrics['parallel_probe_run_records'][2]['seed'] == 44
    assert test_metrics['parallel_probe_run_records'][2]['group_number'] == 2
    assert test_metrics['parallel_probe_run_records'][2]['local_run_number'] == 1
    assert test_metrics['parallel_probe_run_records'][4]['seed'] == 46
    assert test_metrics['parallel_probe_run_records'][4]['group_number'] == 3
    assert test_metrics['parallel_probe_run_records'][4]['local_run_number'] == 1
    assert len(test_metrics['parallel_probe_valid_run_metrics']) == 5
    assert len(test_metrics['parallel_probe_test_run_metrics']) == 5
    assert test_metrics['parallel_probe_best_run_index'] == 2
    assert test_metrics['parallel_probe_best_run_number'] == 3
    assert test_metrics['parallel_probe_best_run_id'] == 'data/model/seed-44'
    assert test_metrics['parallel_probe_best_seed'] == 44
    assert test_metrics['parallel_probe_best_test_loss'] < 0.01
    assert y_pred.shape == (4, 2)
    assert y_true.shape == (4,)


def test_parallel_trainer_uses_budget_derived_group_size(monkeypatch) -> None:
    captured = {'groups': []}

    class FakeAccelerator:
        def free_memory(self):
            pass

    class ChunkedTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs['model']
            self.run_seeds = list(self.model.config.run_seeds)
            captured['groups'].append(self.run_seeds)
            self.accelerator = FakeAccelerator()

        def evaluate(self, dataset):
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': float(len(self.run_seeds))})

        def predict(self, dataset):
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            logits = np.zeros((4, len(self.run_seeds), 2), dtype=np.float32)
            for local_idx, seed in enumerate(self.run_seeds):
                if seed == 44:
                    logits[:, local_idx, :] = np.array(
                        [
                            [5.0, 0.0],
                            [0.0, 5.0],
                            [5.0, 0.0],
                            [0.0, 5.0],
                        ],
                        dtype=np.float32,
                    )
                else:
                    logits[:, local_idx, :] = np.array(
                        [
                            [0.0, 5.0],
                            [5.0, 0.0],
                            [0.0, 5.0],
                            [5.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
            return SimpleNamespace(predictions=logits, label_ids=labels)

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.num_runs = 5
    probe = MLPProbe(
        MLPProbeConfig(
            input_size=mixin.probe_args.input_size,
            hidden_size=mixin.probe_args.hidden_size,
            dropout=mixin.probe_args.dropout,
            num_labels=mixin.probe_args.num_labels,
            n_layers=mixin.probe_args.n_layers,
            task_type=mixin.probe_args.task_type,
            pre_ln=mixin.probe_args.pre_ln,
            use_bias=mixin.probe_args.use_bias,
        )
    )
    bytes_per_run = sum(parameter.numel() for parameter in probe.parameters()) * 4 * 4
    mixin.trainer_args.parallel_probe_training_state_budget_gb = float(bytes_per_run * 2) / float(1024 ** 3)
    monkeypatch.setattr(trainers_module, "Trainer", ChunkedTrainer)

    _model, _valid_metrics, test_metrics, _y_pred, _y_true = mixin._train_parallel_linear_probe_runs(
        train_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
        ],
        valid_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        test_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )

    assert captured['groups'] == [[42, 43], [44, 45], [46]]
    assert test_metrics['parallel_probe_max_group_size'] is None
    assert test_metrics['parallel_probe_training_state_budget_gb'] == pytest.approx(
        mixin.trainer_args.parallel_probe_training_state_budget_gb
    )
    assert test_metrics['parallel_probe_effective_max_group_size'] == 2
    assert test_metrics['parallel_probe_explicit_max_group_size'] is None
    assert test_metrics['parallel_probe_training_state_budget_group_size'] == 2
    assert test_metrics['parallel_probe_estimated_peak_budget_group_size'] is None
    assert test_metrics['parallel_probe_group_size_candidates'] == [2]
    assert test_metrics['parallel_probe_peak_budget_includes_index'] is False
    assert test_metrics['parallel_probe_group_sizes'] == [2, 2, 1]
    assert test_metrics['parallel_probe_trainer_invocations'] == 3
    assert test_metrics['parallel_probe_best_seed'] == 44


def test_parallel_trainer_uses_estimated_peak_budget_derived_group_size(monkeypatch) -> None:
    captured = {'groups': [], 'output_dirs': []}

    class FakeAccelerator:
        def free_memory(self):
            pass

    class ChunkedTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs['model']
            self.run_seeds = list(self.model.config.run_seeds)
            self.accelerator = FakeAccelerator()
            captured['groups'].append(self.run_seeds)
            captured['output_dirs'].append(kwargs['args'].output_dir)

        def evaluate(self, dataset):
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': float(len(self.run_seeds))})

        def predict(self, dataset):
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            logits = np.zeros((4, len(self.run_seeds), 2), dtype=np.float32)
            for local_idx, seed in enumerate(self.run_seeds):
                if seed == 44:
                    logits[:, local_idx, :] = np.array(
                        [
                            [5.0, 0.0],
                            [0.0, 5.0],
                            [5.0, 0.0],
                            [0.0, 5.0],
                        ],
                        dtype=np.float32,
                    )
                else:
                    logits[:, local_idx, :] = np.array(
                        [
                            [0.0, 5.0],
                            [5.0, 0.0],
                            [0.0, 5.0],
                            [5.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
            return SimpleNamespace(predictions=logits, label_ids=labels)

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.num_runs = 5
    probe = MLPProbe(
        MLPProbeConfig(
            input_size=mixin.probe_args.input_size,
            hidden_size=mixin.probe_args.hidden_size,
            dropout=mixin.probe_args.dropout,
            num_labels=mixin.probe_args.num_labels,
            n_layers=mixin.probe_args.n_layers,
            task_type=mixin.probe_args.task_type,
            pre_ln=mixin.probe_args.pre_ln,
            use_bias=mixin.probe_args.use_bias,
        )
    )
    parameter_bytes_per_run = sum(parameter.numel() for parameter in probe.parameters()) * 4 * 4
    activation_count = probes_package.linear_probe_batch_activation_count(
        input_size=mixin.probe_args.input_size,
        hidden_size=mixin.probe_args.hidden_size,
        num_labels=mixin.probe_args.num_labels,
        n_layers=mixin.probe_args.n_layers,
    )
    activation_bytes_per_run = mixin.trainer_args.probe_batch_size * activation_count * 4
    estimated_peak_bytes_per_run = parameter_bytes_per_run + activation_bytes_per_run
    mixin.trainer_args.parallel_probe_estimated_peak_budget_gb = (
        float(estimated_peak_bytes_per_run * 2) / float(1024 ** 3)
    )
    monkeypatch.setattr(trainers_module, "Trainer", ChunkedTrainer)

    _model, _valid_metrics, test_metrics, _y_pred, _y_true = mixin._train_parallel_linear_probe_runs(
        train_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
        ],
        valid_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        test_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )

    assert captured['groups'] == [[42, 43], [44, 45], [46]]
    assert captured['output_dirs'][0].endswith('parallel_group_1')
    assert captured['output_dirs'][1].endswith('parallel_group_2')
    assert captured['output_dirs'][2].endswith('parallel_group_3')
    assert test_metrics['parallel_probe_estimated_peak_budget_gb'] == pytest.approx(
        mixin.trainer_args.parallel_probe_estimated_peak_budget_gb
    )
    assert test_metrics['parallel_probe_effective_max_group_size'] == 2
    assert test_metrics['parallel_probe_explicit_max_group_size'] is None
    assert test_metrics['parallel_probe_training_state_budget_group_size'] is None
    assert test_metrics['parallel_probe_estimated_peak_budget_group_size'] == 2
    assert test_metrics['parallel_probe_group_size_candidates'] == [2]
    assert test_metrics['parallel_probe_peak_budget_includes_index'] is False
    assert test_metrics['parallel_probe_group_sizes'] == [2, 2, 1]
    assert test_metrics['parallel_probe_estimated_peak_group_bytes'] <= estimated_peak_bytes_per_run * 2
    assert test_metrics['parallel_probe_best_seed'] == 44


def test_parallel_trainer_saves_exported_best_probe_when_enabled(monkeypatch) -> None:
    captured = {}

    class FakeAccelerator:
        def free_memory(self):
            pass

    class MetadataTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs['model']
            self.accelerator = FakeAccelerator()

        def evaluate(self, dataset):
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': 8.0})

        def predict(self, dataset):
            logits = np.array(
                [
                    [[3.0, 0.0], [0.0, 3.0]],
                    [[0.0, 3.0], [3.0, 0.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                    [[0.0, 2.0], [2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            return SimpleNamespace(predictions=logits, label_ids=labels)

    def fake_save(**kwargs):
        captured.update(kwargs)

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.save = True
    mixin._save_parallel_best_probe_to_hub = fake_save
    monkeypatch.setattr(trainers_module, "Trainer", MetadataTrainer)

    _model, _valid_metrics, test_metrics, _y_pred, _y_true = mixin._train_parallel_linear_probe_runs(
        train_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
        ],
        valid_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        test_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )

    assert isinstance(captured['best_model'], MLPProbe)
    assert captured['test_metrics'] is test_metrics
    assert captured['test_metrics']['parallel_probe_best_seed'] == 42
    assert captured['log_id'] == 'log'
    assert captured['model_name'] == 'model'
    assert captured['data_name'] == 'data'


def test_parallel_probe_metrics_are_split_by_run() -> None:
    mixin = _trainer_mixin_for_parallel_tests()
    labels = np.array([0, 1, 0, 1])
    logits = np.array(
        [
            [[3.0, 0.0], [0.0, 3.0]],
            [[0.0, 3.0], [3.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    metrics_by_run = mixin._parallel_probe_metrics_by_run(logits, labels, 'fake-data', 'test')

    assert len(metrics_by_run) == 2
    assert metrics_by_run[0]['test_accuracy'] == 1.0
    assert metrics_by_run[1]['test_accuracy'] == 0.0
    assert metrics_by_run[0]['test_loss'] < metrics_by_run[1]['test_loss']


def test_parallel_probe_ensemble_metrics_average_logits() -> None:
    mixin = _trainer_mixin_for_parallel_tests()
    labels = np.array([0, 1, 0, 1])
    logits = np.array(
        [
            [[4.0, 0.0], [2.0, 0.0]],
            [[0.0, 4.0], [0.0, 2.0]],
            [[3.0, 0.0], [1.0, 0.0]],
            [[0.0, 3.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    metrics = mixin._parallel_probe_ensemble_metrics(logits, labels, 'test')
    expected_logits = logits.mean(axis=1)
    expected_loss = F.cross_entropy(torch.tensor(expected_logits), torch.tensor(labels))

    assert metrics['test_accuracy'] == 1.0
    assert metrics['test_loss'] == pytest.approx(float(expected_loss.item()))


def test_seed_ensemble_metrics_from_sequential_run_results() -> None:
    mixin = _trainer_mixin_for_parallel_tests()
    labels = np.array([0, 1, 0, 1])
    run_0_logits = np.array(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    run_1_logits = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    run_results = [
        (0, 0.2, run_0_logits, labels, 42, None),
        (1, 0.3, run_1_logits, labels, 43, None),
    ]

    metrics = mixin._seed_ensemble_metrics_from_run_results(run_results, 'test')
    expected_logits = np.stack([run_0_logits, run_1_logits], axis=1).mean(axis=1)
    expected_loss = F.cross_entropy(torch.tensor(expected_logits), torch.tensor(labels))

    assert metrics['test_accuracy'] == 1.0
    assert metrics['test_loss'] == pytest.approx(float(expected_loss.item()))


def test_seed_ensemble_metrics_reject_label_mismatch() -> None:
    mixin = _trainer_mixin_for_parallel_tests()
    run_results = [
        (0, 0.2, np.zeros((2, 2), dtype=np.float32), np.array([0, 1]), 42, None),
        (1, 0.3, np.zeros((2, 2), dtype=np.float32), np.array([1, 0]), 43, None),
    ]

    with pytest.raises(AssertionError, match="labels changed"):
        mixin._seed_ensemble_metrics_from_run_results(run_results, 'test')


def test_parallel_probe_ensemble_probability_mode_returns_metric_compatible_scores() -> None:
    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.parallel_probe_ensemble_average_mode = 'probabilities'
    logits = np.array(
        [
            [[4.0, 0.0], [0.0, 1.0]],
            [[0.0, 3.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    scores = mixin._parallel_probe_ensemble_predictions(logits, 'singlelabel')
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    expected_probabilities = (exp_logits / exp_logits.sum(axis=-1, keepdims=True)).mean(axis=1)

    assert np.allclose(np.exp(scores), expected_probabilities, atol=1e-6)


def test_parallel_trainer_reports_chunked_ensemble_metrics(monkeypatch) -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    def logits_for_seed(seed: int) -> np.ndarray:
        scale = 1.0 + (float(seed - 42) * 0.25)
        return np.array(
            [
                [4.0 * scale, 0.0],
                [0.0, 4.0 * scale],
                [3.0 * scale, 0.0],
                [0.0, 3.0 * scale],
            ],
            dtype=np.float32,
        )

    class FakeAccelerator:
        def free_memory(self):
            pass

    class ChunkedTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs['model']
            self.run_seeds = list(self.model.config.run_seeds)
            self.accelerator = FakeAccelerator()

        def evaluate(self, dataset):
            return {'eval_loss': 0.0}

        def train(self):
            return SimpleNamespace(metrics={'train_runtime': float(len(self.run_seeds))})

        def predict(self, dataset):
            logits = np.stack(
                [logits_for_seed(seed) for seed in self.run_seeds],
                axis=1,
            )
            return SimpleNamespace(predictions=logits, label_ids=labels)

    mixin = _trainer_mixin_for_parallel_tests()
    mixin.trainer_args.num_runs = 4
    mixin.trainer_args.parallel_probe_max_group_size = 2
    monkeypatch.setattr(trainers_module, "Trainer", ChunkedTrainer)

    _model, valid_metrics, test_metrics, _y_pred, _y_true = mixin._train_parallel_linear_probe_runs(
        train_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
        ],
        valid_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        test_dataset=[
            (torch.zeros(4), torch.tensor(0)),
            (torch.ones(4), torch.tensor(1)),
            (torch.full((4,), 2.0), torch.tensor(0)),
            (torch.full((4,), 3.0), torch.tensor(1)),
        ],
        data_collator=None,
        tokenizer=None,
        log_id='log',
        model_name='model',
        data_name='data',
    )
    expected_logits = np.stack(
        [logits_for_seed(seed) for seed in (42, 43, 44, 45)],
        axis=1,
    ).mean(axis=1)
    expected_loss = F.cross_entropy(torch.tensor(expected_logits), torch.tensor(labels))

    assert valid_metrics['parallel_probe_ensemble_eval_accuracy'] == 1.0
    assert test_metrics['parallel_probe_ensemble_test_accuracy'] == 1.0
    assert test_metrics['parallel_probe_ensemble_test_loss'] == pytest.approx(float(expected_loss.item()))
    assert test_metrics['parallel_probe_ensemble_average_mode'] == 'logits'
    assert test_metrics['parallel_probe_group_sizes'] == [2, 2]


def test_parallel_probe_metrics_use_run_specific_singlelabel_labels() -> None:
    mixin = _trainer_mixin_for_parallel_tests()
    labels = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ]
    )
    logits = np.array(
        [
            [[3.0, 0.0], [0.0, 3.0]],
            [[0.0, 3.0], [3.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    metrics_by_run = mixin._parallel_probe_metrics_by_run(logits, labels, 'fake-data', 'test')

    assert len(metrics_by_run) == 2
    assert metrics_by_run[0]['test_accuracy'] == 1.0
    assert metrics_by_run[1]['test_accuracy'] == 1.0
    assert metrics_by_run[0]['test_loss'] == pytest.approx(metrics_by_run[1]['test_loss'])


def test_parallel_probe_loss_reporting_uses_run_specific_scalar_labels() -> None:
    mixin = _trainer_mixin_for_parallel_tests(task_type='regression')
    logits = np.array(
        [
            [[0.0], [10.0]],
            [[1.0], [11.0]],
            [[2.0], [12.0]],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
        ],
        dtype=np.float32,
    )

    losses = mixin._parallel_probe_losses_by_run(logits, labels, 'regression')

    assert losses == [0.0, 0.0]


def test_eval_accumulation_accounts_for_parallel_output_multiplier(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    without_multiplier = _compute_eval_accumulation_steps(
        eval_dataset_size=3_000_000,
        batch_size=1_000,
        num_labels=10,
        task_type='singlelabel',
        output_multiplier=1,
    )
    with_multiplier = _compute_eval_accumulation_steps(
        eval_dataset_size=3_000_000,
        batch_size=1_000,
        num_labels=10,
        task_type='singlelabel',
        output_multiplier=8,
    )

    assert without_multiplier is None
    assert with_multiplier is not None
    assert with_multiplier > 0
