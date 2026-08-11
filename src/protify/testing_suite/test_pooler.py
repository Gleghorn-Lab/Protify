import torch
import pytest

try:
    from src.protify.pooler import Pooler
except ImportError:
    try:
        from protify.pooler import Pooler
    except ImportError:
        from ..pooler import Pooler


batch_size = 2  # b
sequence_length = 4  # l
hidden_size = 8  # d


@pytest.fixture
def emb() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, sequence_length, hidden_size)  # (b, l, d)


@pytest.fixture
def mask() -> torch.Tensor:
    # first sample: 3 real tokens, second: all 4
    return torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.float32,
    )  # (b, l)


@pytest.fixture
def attentions() -> torch.Tensor:
    torch.manual_seed(0)
    num_attention_layers = 2  # a
    return torch.randn(
        batch_size,
        num_attention_layers,
        sequence_length,
        sequence_length,
    ).abs()  # (b, a, l, l)


def test_mean_no_mask(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['mean'])
    pooled = pooler.mean_pooling(emb)  # (b, d)
    expected = emb.mean(dim=1)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    assert torch.allclose(pooled, expected)


def test_mean_with_mask(emb: torch.Tensor, mask: torch.Tensor) -> None:
    # emb: (b, l, d); mask: (b, l)
    pooler = Pooler(['mean'])
    pooled = pooler.mean_pooling(emb, attention_mask=mask)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    # Manual check for first sample: mean of first 3 tokens
    first_sequence_mean = emb[0, :3, :].mean(dim=0)  # (d,)
    assert torch.allclose(pooled[0], first_sequence_mean, atol=1e-6)  # both (d,)


def test_max_no_mask(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['max'])
    pooled = pooler.max_pooling(emb)  # (b, d)
    expected = emb.max(dim=1).values  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    assert torch.allclose(pooled, expected)


def test_max_with_mask(emb: torch.Tensor, mask: torch.Tensor) -> None:
    # emb: (b, l, d); mask: (b, l)
    pooler = Pooler(['max'])
    pooled = pooler.max_pooling(emb, attention_mask=mask)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)


def test_max_with_mask_negative_values() -> None:
    """Masked positions must not win when all unmasked values are negative."""
    torch.manual_seed(0)
    embeddings = torch.full((1, 4, hidden_size), -5.0)  # (1, l=4, d)
    embeddings[0, 0, :] = -1.0  # slice: (d,); embeddings remains (1, l, d)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)  # (1, l)
    pooler = Pooler(['max'])
    pooled = pooler.max_pooling(embeddings, attention_mask=attention_mask)  # (1, d)
    # Max of unmasked positions should be -1.0, not 0.0 from masked
    expected = torch.full((1, hidden_size), -1.0)  # (1, d)
    assert torch.allclose(pooled, expected)


def test_norm_no_mask(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['norm'])
    pooled = pooler.norm_pooling(emb)  # (b, d)
    expected = emb.norm(dim=1, p=2)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    assert torch.allclose(pooled, expected)


def test_norm_with_mask(emb: torch.Tensor, mask: torch.Tensor) -> None:
    # emb: (b, l, d); mask: (b, l)
    pooler = Pooler(['norm'])
    pooled = pooler.norm_pooling(emb, attention_mask=mask)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)


def test_median_no_mask(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['median'])
    pooled = pooler.median_pooling(emb)  # (b, d)
    expected = emb.median(dim=1).values  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    assert torch.allclose(pooled, expected)


def test_var_no_mask(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['var'])
    pooled = pooler.var_pooling(emb)  # (b, d)
    expected = emb.var(dim=1)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    assert torch.allclose(pooled, expected)


def test_var_with_mask(emb: torch.Tensor, mask: torch.Tensor) -> None:
    # emb: (b, l, d); mask: (b, l)
    pooler = Pooler(['var'])
    pooled = pooler.var_pooling(emb, attention_mask=mask)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    # Manual variance for first sample over 3 unmasked tokens (population variance)
    unmasked = emb[0, :3, :]  # (3, d)
    mean = unmasked.mean(dim=0)  # (d,)
    manual_variance = ((unmasked - mean) ** 2).mean(dim=0)  # (d,)
    assert torch.allclose(pooled[0], manual_variance, atol=1e-6)  # both (d,)


def test_std_equals_sqrt_var_with_mask(
    emb: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    # emb: (b, l, d); mask: (b, l)
    pooler = Pooler(['std'])
    standard_deviation = pooler.std_pooling(emb, attention_mask=mask)  # (b, d)
    variance = pooler.var_pooling(emb, attention_mask=mask)  # (b, d)
    expected_standard_deviation = torch.sqrt(variance)  # (b, d)
    assert torch.allclose(standard_deviation, expected_standard_deviation, atol=1e-6)


def test_cls_returns_first_token(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['cls'])
    pooled = pooler.cls_pooling(emb)  # (b, d)
    first_token_embeddings = emb[:, 0, :]  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)
    assert torch.allclose(pooled, first_token_embeddings)


def test_cls_ignores_mask(emb: torch.Tensor, mask: torch.Tensor) -> None:
    # emb: (b, l, d); mask: (b, l)
    pooler = Pooler(['cls'])
    pooled_without_mask = pooler.cls_pooling(emb)  # (b, d)
    pooled_with_mask = pooler.cls_pooling(emb, attention_mask=mask)  # (b, d)
    assert torch.allclose(pooled_without_mask, pooled_with_mask)


def test_parti_shape(
    emb: torch.Tensor,
    attentions: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    # emb: (b, l, d); attentions: (b, a, l, l); mask: (b, l)
    pooler = Pooler(['parti'])
    pooled = pooler._pool_parti(emb, attentions, attention_mask=mask)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)


def test_call_single_type(emb: torch.Tensor) -> None:
    # emb: (b, l, d)
    pooler = Pooler(['mean'])
    pooled = pooler(emb)  # (b, d)
    assert pooled.shape == (batch_size, hidden_size)


def test_call_multiple_types_concat(
    emb: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    # emb: (b, l, d); mask: (b, l)
    types = ['mean', 'max', 'cls']
    pooler = Pooler(types)
    pooled = pooler(emb, attention_mask=mask)  # (b, 3d)
    assert pooled.shape == (batch_size, len(types) * hidden_size)


def test_unknown_pooling_type_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown pooling types"):
        Pooler(['mean', 'nonexistent'])


def test_repeated_pooling_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="appear more than once"):
        Pooler(['mean', 'max', 'mean'])


def test_std_pooling_agrees_with_and_without_a_mask(emb: torch.Tensor) -> None:
    # emb: (b, l, d). A full mask must give the same answer as no mask at all, which
    # needs both branches to use the population standard deviation.
    pooler = Pooler(['std'])
    full_mask = torch.ones(emb.shape[0], emb.shape[1])  # (b, l)

    assert torch.allclose(
        pooler(emb),  # (b, d)
        pooler(emb, attention_mask=full_mask),  # (b, d)
        atol=1e-5,
    )
