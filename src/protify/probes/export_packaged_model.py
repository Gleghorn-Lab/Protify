import shutil
import tempfile

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from torch import nn

try:
    from base_models.supported_models import all_presets_with_paths
    from probes.hybrid_probe import HybridProbe
    from probes.packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
    from utils import print_message
except ImportError:
    from ..base_models.supported_models import all_presets_with_paths
    from .hybrid_probe import HybridProbe
    from .packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
    from ..utils import print_message


def _infer_probe_type(probe_model: nn.Module) -> str:
    probe_class_name = probe_model.__class__.__name__
    if probe_class_name == "MLPProbe":
        return "mlp"
    if probe_class_name in ["TransformerForSequenceClassification", "TransformerForTokenClassification"]:
        return "transformer"
    if probe_class_name in ["LyraForSequenceClassification", "LyraForTokenClassification"]:
        return "lyra"
    raise ValueError(f"Unsupported probe class for packaged export: {probe_class_name}")


def _is_supported_base_model(source_model_name: str) -> bool:
    if source_model_name not in all_presets_with_paths:
        return False
    model_name_l = source_model_name.lower()
    if "random" in model_name_l:
        return False
    if "onehot" in model_name_l:
        return False
    return True


def _extract_sep_token_id(tokenizer: Any) -> int | None:
    try:
        tokenizer_backend = tokenizer.tokenizer
    except AttributeError:
        tokenizer_backend = tokenizer
    if tokenizer_backend.sep_token_id is not None:
        return int(tokenizer_backend.sep_token_id)
    if tokenizer_backend.eos_token_id is not None:
        return int(tokenizer_backend.eos_token_id)
    return None


def _copy_runtime_code(export_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_package_dir = repo_root / "src" / "protify"
    dst_package_dir = export_dir / "protify"
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

    packaged_model_file = Path(__file__).with_name("packaged_probe_model.py")
    shutil.copy2(packaged_model_file, export_dir / "packaged_probe_model.py")


def _build_packaged_model(
    trained_model: nn.Module,
    source_model_name: str,
    probe_args: Any,
    embedding_args: Any,
    tokenizer: Any,
    ppi: bool,
) -> PackagedProbeModel:
    if isinstance(trained_model, HybridProbe):
        base_model = trained_model.model
        probe_model = trained_model.probe
    else:
        base_model = None
        probe_model = trained_model

    probe_type = _infer_probe_type(probe_model)
    probe_config_dict = probe_model.config.to_dict()
    sep_token_id = _extract_sep_token_id(tokenizer)
    packaged_config = PackagedProbeConfig(
        base_model_name=source_model_name,
        probe_type=probe_type,
        probe_config=probe_config_dict,
        tokenwise=probe_args.tokenwise,
        matrix_embed=embedding_args.matrix_embed,
        pooling_types=embedding_args.pooling_types,
        task_type=probe_args.task_type,
        num_labels=probe_args.num_labels,
        ppi=ppi,
        add_token_ids=probe_args.add_token_ids,
        sep_token_id=sep_token_id,
    )
    packaged_model = PackagedProbeModel(config=packaged_config, base_model=base_model, probe=probe_model)
    return packaged_model.cpu()


def export_packaged_model_to_hub(
    trained_model: nn.Module,
    source_model_name: str,
    probe_args: Any,
    embedding_args: Any,
    tokenizer: Any,
    repo_id: str,
    model_card: str,
    ppi: bool = False,
    private: bool = True,
    hf_token: str | None = None,
) -> tuple[bool, str]:
    if not _is_supported_base_model(source_model_name):
        return False, f"Packaged export is not supported for base model: {source_model_name}"

    packaged_model = _build_packaged_model(
        trained_model=trained_model,
        source_model_name=source_model_name,
        probe_args=probe_args,
        embedding_args=embedding_args,
        tokenizer=tokenizer,
        ppi=ppi,
    )

    with tempfile.TemporaryDirectory(prefix="protify_packaged_model_") as temp_dir:
        export_dir = Path(temp_dir)

        packaged_model.config.auto_map = {
            "AutoConfig": "packaged_probe_model.PackagedProbeConfig",
            "AutoModel": "packaged_probe_model.PackagedProbeModel",
        }
        packaged_model.config.architectures = ["PackagedProbeModel"]
        packaged_model.save_pretrained(str(export_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(export_dir))
        _copy_runtime_code(export_dir)
        readme_path = export_dir / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")

        if hf_token is None:
            api = HfApi()
        else:
            api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            path_in_repo="",
        )

    print_message(f"Packaged model and tokenizer uploaded to Hugging Face Hub: {repo_id}")
    return True, f"Uploaded packaged model to {repo_id}"
