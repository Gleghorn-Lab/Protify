import gc
import shutil
import tempfile
import torch
from pathlib import Path
from transformers import AutoModel, BertConfig, BertModel, BertTokenizerFast

try:
    from probes.mlp_probe import MLPProbe, MLPProbeConfig
    from probes.packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
    from probes.transformer_probe import TransformerForSequenceClassification, TransformerProbeConfig
except ImportError:
    from ..probes.mlp_probe import MLPProbe, MLPProbeConfig
    from ..probes.packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
    from ..probes.transformer_probe import TransformerForSequenceClassification, TransformerProbeConfig


def _copy_runtime_code(save_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_package_dir = repo_root / "src" / "protify"
    dst_package_dir = save_dir / "protify"
    excluded_directories = {
        ".cache",
        "__pycache__",
        "benchmarks",
        "fastplms",
        "logs",
        "scripts",
        "testing_suite",
    }
    for src_file in src_package_dir.rglob("*.py"):
        relative_path = src_file.relative_to(src_package_dir)
        if any(part in excluded_directories for part in relative_path.parts[:-1]):
            continue
        if not src_file.is_file():
            continue
        dst_file = dst_package_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    # Also copy packaged_probe_model.py to save_dir root as a standalone
    # version with absolute imports so transformers dynamic_module_utils can
    # resolve the import chain without relative path traversal.
    packaged_model_file = repo_root / "src" / "protify" / "probes" / "packaged_probe_model.py"
    standalone = save_dir / "packaged_probe_model.py"
    content = packaged_model_file.read_text(encoding="utf-8")
    content = content.replace("from ..base_models.supported_models import", "from protify.base_models.supported_models import")
    content = content.replace("from ..pooler import", "from protify.pooler import")
    content = content.replace("from .get_probe import", "from protify.probes.get_probe import")
    standalone.write_text(content, encoding="utf-8")


def _create_tiny_backbone(backbone_dir: Path) -> tuple[BertModel, BertTokenizerFast]:
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "B", "C", "D"]
    vocab_path = backbone_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = BertModel(config).eval()
    model.save_pretrained(str(backbone_dir))
    tokenizer.save_pretrained(str(backbone_dir))
    return model, tokenizer


def _save_and_load_with_automodel(
    packaged_model: PackagedProbeModel,
    tokenizer: BertTokenizerFast,
    model_dir: Path,
) -> AutoModel:
    packaged_model.config.auto_map = {
        "AutoConfig": "packaged_probe_model.PackagedProbeConfig",
        "AutoModel": "packaged_probe_model.PackagedProbeModel",
    }
    packaged_model.config.architectures = ["PackagedProbeModel"]
    packaged_model.save_pretrained(str(model_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(model_dir))
    _copy_runtime_code(model_dir)
    return AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)


def test_linear_packaged_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix="protify_linear_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "linear_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = MLPProbeConfig(
            input_size=16,
            hidden_size=32,
            dropout=0.1,
            num_labels=3,
            n_layers=1,
            task_type="singlelabel",
        )
        probe = MLPProbe(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="mlp",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=False,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=3,
            ppi=False,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C A", "B C D A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        outputs = loaded_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )  # logits: (b=2, c=3)
        assert outputs.logits.shape == (2, 3), f"Unexpected linear packaged logits shape: {outputs.logits.shape}"
        del loaded_model
        gc.collect()


def test_transformer_packaged_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix="protify_transformer_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "transformer_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = TransformerProbeConfig(
            input_size=16,
            hidden_size=16,
            classifier_size=24,
            transformer_dropout=0.1,
            classifier_dropout=0.1,
            num_labels=2,
            n_layers=1,
            head_size=8,
            task_type="singlelabel",
            rotary=False,
            attention_backend="sdpa",
            output_s_max=False,
            pre_ln=True,
            probe_pooling_types=["mean"],
            use_bias=False,
            add_token_ids=False,
        )
        probe = TransformerForSequenceClassification(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="transformer",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=True,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=2,
            ppi=False,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C D", "D C B A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        outputs = loaded_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )  # logits: (b=2, c=2)
        assert outputs.logits.shape == (2, 2), f"Unexpected transformer packaged logits shape: {outputs.logits.shape}"
        del loaded_model
        gc.collect()


def test_ppi_packaged_inference_with_and_without_token_type_ids() -> None:
    with tempfile.TemporaryDirectory(prefix="protify_ppi_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "ppi_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = MLPProbeConfig(
            input_size=32,
            hidden_size=24,
            dropout=0.1,
            num_labels=2,
            n_layers=1,
            task_type="singlelabel",
        )
        probe = MLPProbe(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="mlp",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=False,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=2,
            ppi=True,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        pair_batch = tokenizer(
            ["A B C", "B C D"],
            ["D C B", "A C B"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l_pair=9)

        outputs_with_token_types = loaded_model(
            input_ids=pair_batch["input_ids"],
            attention_mask=pair_batch["attention_mask"],
            token_type_ids=pair_batch["token_type_ids"],
        )  # logits: (b=2, c=2)
        assert outputs_with_token_types.logits.shape == (2, 2), "PPI logits shape mismatch with token_type_ids"

        outputs_without_token_types = loaded_model(
            input_ids=pair_batch["input_ids"],
            attention_mask=pair_batch["attention_mask"],
        )  # logits: (b=2, c=2)
        assert outputs_without_token_types.logits.shape == (2, 2), "PPI logits shape mismatch without token_type_ids"
        del loaded_model
        gc.collect()


def main() -> None:
    torch.manual_seed(0)
    test_linear_packaged_roundtrip()
    test_transformer_packaged_roundtrip()
    test_ppi_packaged_inference_with_and_without_token_type_ids()
    print("Packaged probe model smoke tests passed.")


if __name__ == "__main__":
    main()
