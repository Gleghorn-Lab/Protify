import networkx as nx
import numpy as np
import torch
from typing import Callable, Dict, List, Optional


class Pooler:
    def __init__(self, pooling_types: List[str]) -> None:
        assert 'bom' not in pooling_types, (
            "'bom' pooling is only supported by the Transformer probe "
            "(implemented in probes/transformer_probe.py). Pass --probe_type transformer "
            "with --probe_pooling_types bom, and do not include 'bom' in --embedding_pooling_types."
        )
        self.pooling_types = pooling_types
        self.pooling_options: Dict[str, Callable[..., torch.Tensor]] = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'norm': self.norm_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
            'parti': self._pool_parti,
        }
        # Reject bad names here rather than at the first forward pass, where the run has
        # already paid for a model load and an embedding pass.
        unknown = [name for name in pooling_types if name not in self.pooling_options]
        if unknown:
            raise ValueError(
                f"Unknown pooling types {unknown}; expected any of {sorted(self.pooling_options)}."
            )
        duplicates = [name for name in set(pooling_types) if pooling_types.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Pooling types {duplicates} appear more than once. Each one widens the "
                f"embedding by the hidden size, so repeats only waste space."
            )

    def _create_pooled_matrices_across_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        # attentions: (b, r, l, l); r = attention matrices or layers
        maxed_attentions = torch.max(attentions, dim=1)[0]  # (b, l, l)
        return maxed_attentions  # (b, l, l)

    def _page_rank(
        self,
        attention_matrix: np.ndarray,
        personalization: Optional[Dict[int, float]] = None,
        nstart: Optional[Dict[int, float]] = None,
        prune_type: str = "top_k_outdegree",
    ) -> Dict[int, float]:
        # attention_matrix: (l, l). The compatibility-only prune_type is intentionally unused.
        graph = self._convert_to_graph(attention_matrix)
        if graph.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(
                "The number of nodes in the graph should be equal to the number of tokens in sequence! "
                f"You have {graph.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens."
            )
        if graph.number_of_edges() == 0:
            raise Exception("You don't seem to have any attention edges left in the graph.")

        return nx.pagerank(
            graph,
            alpha=0.85,
            tol=1e-06,
            weight='weight',
            personalization=personalization,
            nstart=nstart,
            max_iter=100,
        )

    def _convert_to_graph(self, matrix: np.ndarray) -> nx.DiGraph:
        # matrix: (l, l)
        graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        return graph

    def _calculate_importance_weights(
        self,
        dict_importance: Dict[int, float],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        # attention_mask: (l,); l_valid = number of retained nodes
        if attention_mask is not None:
            for k in list(dict_importance.keys()):
                if attention_mask[k] == 0:
                    del dict_importance[k]

        total = sum(dict_importance.values())
        return np.array([v / total for _, v in dict_importance.items()])  # (l_valid,)

    def _pool_parti(
        self,
        emb: torch.Tensor,
        attentions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # emb: (b, l, d); attentions: (b, r, l, l); attention_mask: (b, l)
        # r = attention matrices or layers; l_valid = unmasked tokens in one sample
        # The historical NumPy conversion requires CPU tensors without gradients.
        maxed_attentions = self._create_pooled_matrices_across_layers(attentions).numpy()  # (b, l, l)
        emb_pooled: List[np.ndarray] = []
        for e, a, mask in zip(emb, maxed_attentions, attention_mask):
            # e: (l, d); a: (l, l); mask: (l,)
            dict_importance = self._page_rank(a)
            importance_weights = self._calculate_importance_weights(dict_importance, mask)  # (l_valid,)
            num_tokens = int(mask.sum().item())  # l_valid
            pooled_embedding = np.average(  # (d,)
                e[:num_tokens],  # (l_valid, d)
                weights=importance_weights,
                axis=0,
            )
            emb_pooled.append(pooled_embedding)
        pooled = torch.tensor(np.array(emb_pooled))  # (b, d)
        return pooled  # (b, d)

    def mean_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l)
        if attention_mask is None:
            return emb.mean(dim=1)  # (b, d)

        attention_mask = attention_mask.unsqueeze(-1)  # (b, l, 1)
        return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)

    def max_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l)
        if attention_mask is None:
            return emb.max(dim=1).values  # (b, d)

        mask = attention_mask.unsqueeze(-1).bool()  # (b, l, 1)
        return emb.masked_fill(~mask, float('-inf')).max(dim=1).values  # (b, d)

    def norm_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l)
        if attention_mask is None:
            return emb.norm(dim=1, p=2)  # (b, d)

        attention_mask = attention_mask.unsqueeze(-1)  # (b, l, 1)
        return (emb * attention_mask).norm(dim=1, p=2)  # (b, d)

    def median_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l)
        if attention_mask is None:
            return emb.median(dim=1).values  # (b, d)

        mask = attention_mask.bool()  # (b, l); l_i = valid tokens in sample i
        results: List[torch.Tensor] = []
        for i in range(emb.shape[0]):
            valid = emb[i, mask[i]]  # (l_i, d)
            results.append(valid.median(dim=0).values)  # (d,)
        return torch.stack(results)  # (b, d)

    def std_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l)
        # Population standard deviation in both branches. The masked path divides by the
        # residue count, so an unbiased estimate here would not agree with it.
        if attention_mask is None:
            return emb.std(dim=1, unbiased=False)  # (b, d)

        var = self.var_pooling(emb, attention_mask, **kwargs)  # (b, d)
        return torch.sqrt(var)  # (b, d)

    def var_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l)
        # Population variance in both branches, matching std_pooling. The masked path
        # divides by the residue count, so an unbiased estimate here would disagree with
        # it, and with std ** 2, for the same sequence.
        if attention_mask is None:
            return emb.var(dim=1, unbiased=False)  # (b, d)

        attention_mask = attention_mask.unsqueeze(-1)  # (b, l, 1)
        mean = (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)
        mean = mean.unsqueeze(1)  # (b, 1, d)
        squared_diff = (emb - mean) ** 2  # (b, l, d)
        var = (squared_diff * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)
        return var  # (b, d)

    def cls_pooling(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # emb: (b, l, d)
        return emb[:, 0, :]  # (b, d)

    def __call__(
        self,
        emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attentions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l); attentions: (b, r, l, l)
        # r = attention matrices or layers; p = number of configured pooling operations
        if attention_mask is not None:
            assert attention_mask.sum(dim=-1).min() > 0, (
                "Pooler received samples with all-zero attention masks. "
                "This causes NaN from division by zero. Filter empty inputs before pooling."
            )
        final_emb: List[torch.Tensor] = []
        for pooling_type in self.pooling_types:
            pooled_embedding = self.pooling_options[pooling_type](
                emb=emb,
                attention_mask=attention_mask,
                attentions=attentions,
            )  # (b, d)
            final_emb.append(pooled_embedding)
        return torch.cat(final_emb, dim=-1)  # (b, p * d)


if __name__ == "__main__":
    pooler = Pooler(pooling_types=['max', 'parti'])

    batch_size = 8  # b
    seq_len = 64  # l
    hidden_size = 128  # d
    num_layers = 12  # r
    emb = torch.randn(batch_size, seq_len, hidden_size)  # (b, l, d)
    attentions = torch.randn(batch_size, num_layers, seq_len, seq_len)  # (b, r, l, l)
    attention_mask = torch.ones(batch_size, seq_len)  # (b, l)

    y = pooler(emb=emb, attention_mask=attention_mask, attentions=attentions)  # (b, 2 * d)
    print(y.shape)
