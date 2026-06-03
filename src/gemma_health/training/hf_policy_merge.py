from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


ModelDtype = Literal["bfloat16", "float16", "float32", "auto"]


@dataclass(frozen=True)
class HfPolicyAdapterMergeConfig:
    base_model_id: str
    adapter_model_id: str
    output_dir: Path
    hub_model_id: str | None
    private: bool
    overwrite: bool
    dtype: ModelDtype
    max_shard_size: str
    safe_merge: bool
    missing_target_policy: Literal["error", "warn"]


@dataclass(frozen=True)
class HfPolicyAdapterMergeResult:
    base_model_id: str
    adapter_model_id: str
    output_dir: str
    hub_model_id: str | None
    dtype: str
    safe_merge: bool
    skipped_lora_modules: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def merge_policy_adapter_with_hf_peft(merge_config: HfPolicyAdapterMergeConfig) -> HfPolicyAdapterMergeResult:
    _prepare_output_dir(merge_config.output_dir, overwrite=merge_config.overwrite)

    try:
        import torch
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    except ImportError as err:
        raise RuntimeError("Install torch, transformers, safetensors, and huggingface_hub before HF merging") from err

    dtype = _torch_dtype(torch, merge_config.dtype)
    print(f"loading base model={merge_config.base_model_id}", flush=True)
    model = _load_base_model(
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        merge_config.base_model_id,
        dtype=dtype,
    )
    print(f"downloading policy adapter={merge_config.adapter_model_id}", flush=True)
    adapter_path = Path(snapshot_download(merge_config.adapter_model_id, repo_type="model"))
    adapter_state = load_file(adapter_path / "adapter_model.safetensors")
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    print(f"merging LoRA tensors safe_merge={merge_config.safe_merge}", flush=True)
    applied, skipped = _merge_lora_state_into_model(
        model,
        adapter_state,
        adapter_config,
        torch_module=torch,
        safe_merge=merge_config.safe_merge,
        missing_target_policy=merge_config.missing_target_policy,
    )
    print(f"merged LoRA modules={applied} skipped={skipped}", flush=True)
    model.eval()

    processor = _load_processor_or_tokenizer(AutoProcessor, AutoTokenizer, merge_config.base_model_id)
    _align_special_tokens(model, processor)

    print(f"saving HF merged model to {merge_config.output_dir}", flush=True)
    model.save_pretrained(
        merge_config.output_dir,
        safe_serialization=True,
        max_shard_size=merge_config.max_shard_size,
    )
    processor.save_pretrained(merge_config.output_dir)
    repaired = _repair_missing_base_tensors(
        base_model_id=merge_config.base_model_id,
        output_dir=merge_config.output_dir,
        hf_hub_download=hf_hub_download,
        safe_open=safe_open,
        save_file=save_file,
    )
    print(f"repaired missing base tensors={repaired}", flush=True)
    _write_model_card(merge_config)

    if merge_config.hub_model_id is not None:
        print(f"uploading HF merged model to {merge_config.hub_model_id}", flush=True)
        api = HfApi()
        api.create_repo(
            repo_id=merge_config.hub_model_id,
            repo_type="model",
            private=merge_config.private,
            exist_ok=True,
        )
        _upload_large_folder(api, merge_config)

    del model
    del processor

    return HfPolicyAdapterMergeResult(
        base_model_id=merge_config.base_model_id,
        adapter_model_id=merge_config.adapter_model_id,
        output_dir=str(merge_config.output_dir),
        hub_model_id=merge_config.hub_model_id,
        dtype=merge_config.dtype,
        safe_merge=merge_config.safe_merge,
        skipped_lora_modules=skipped,
    )


def _torch_dtype(torch: Any, dtype: ModelDtype) -> Any:
    if dtype == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _load_base_model(
    image_text_model_cls: Any,
    causal_model_cls: Any,
    base_model_id: str,
    *,
    dtype: Any,
) -> Any:
    kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if dtype == "auto":
        kwargs["dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
    try:
        return image_text_model_cls.from_pretrained(base_model_id, **kwargs)
    except Exception as image_text_err:
        try:
            return causal_model_cls.from_pretrained(base_model_id, **kwargs)
        except Exception as causal_err:
            raise RuntimeError(
                f"failed to load {base_model_id} with AutoModelForImageTextToText or AutoModelForCausalLM"
            ) from causal_err


def _load_processor_or_tokenizer(processor_cls: Any, tokenizer_cls: Any, base_model_id: str) -> Any:
    try:
        return processor_cls.from_pretrained(base_model_id, trust_remote_code=True)
    except Exception:
        return tokenizer_cls.from_pretrained(base_model_id, trust_remote_code=True)


def _align_special_tokens(model: Any, processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    for token_name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, token_name, None)
        if token_id is None:
            continue
        setattr(model.config, token_name, token_id)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, token_name, token_id)


def _merge_lora_state_into_model(
    model: Any,
    adapter_state: dict[str, Any],
    adapter_config: dict[str, Any],
    *,
    torch_module: Any,
    safe_merge: bool,
    missing_target_policy: Literal["error", "warn"],
) -> tuple[int, int]:
    rank_pattern = adapter_config.get("rank_pattern") or {}
    alpha_pattern = adapter_config.get("alpha_pattern") or {}
    default_rank = int(adapter_config["r"])
    default_alpha = float(adapter_config["lora_alpha"])
    applied = 0
    skipped = 0
    model_state = model.state_dict()

    with torch_module.no_grad():
        for lora_a_key, lora_a in adapter_state.items():
            if ".lora_A." not in lora_a_key or not lora_a_key.endswith(".weight"):
                continue
            lora_b_key = lora_a_key.replace(".lora_A.", ".lora_B.", 1)
            if lora_b_key not in adapter_state:
                raise ValueError(f"missing LoRA B tensor for {lora_a_key}")
            module_name = _adapter_module_name(lora_a_key)
            try:
                weight = _find_target_weight_tensor(model_state, module_name)
            except ValueError as err:
                if missing_target_policy == "error":
                    raise
                skipped += 1
                print(f"skipping unmatched LoRA target={module_name}: {err}", flush=True)
                continue
            lora_b = adapter_state[lora_b_key]
            rank = int(rank_pattern.get(module_name, default_rank))
            alpha = float(alpha_pattern.get(module_name, default_alpha))
            delta = (lora_b.float() @ lora_a.float()) * (alpha / rank)
            if delta.shape != weight.shape:
                raise ValueError(f"LoRA delta shape mismatch for {module_name}: {delta.shape} != {weight.shape}")
            if safe_merge and not torch_module.isfinite(delta).all():
                raise ValueError(f"LoRA delta has non-finite values for {module_name}")
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
            applied += 1

    if applied == 0:
        raise ValueError("adapter contains no mergeable LoRA linear tensors")
    return applied, skipped


def _adapter_module_name(lora_key: str) -> str:
    module_name = lora_key.split(".lora_A.", maxsplit=1)[0]
    for prefix in ("base_model.model.", "model."):
        if module_name.startswith(prefix):
            return module_name.removeprefix(prefix)
    return module_name


def _find_target_weight_tensor(model_state: dict[str, Any], module_name: str) -> Any:
    direct_candidates = []
    for candidate in _module_name_candidates(module_name):
        direct_candidates.extend((f"{candidate}.weight", f"{candidate}.linear.weight"))
    for candidate in direct_candidates:
        if candidate in model_state:
            return model_state[candidate]

    normalized_target = _normalized_module_name(module_name)
    matches = [
        tensor
        for name, tensor in model_state.items()
        if name.endswith(".weight") and _normalized_weight_name(name) == normalized_target
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous target weight for adapter tensor {module_name}: {len(matches)} matches")
    suffix_matches = [
        tensor
        for name, tensor in model_state.items()
        if name.endswith(".weight") and normalized_target in _normalized_weight_name(name)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(f"ambiguous suffix target weight for adapter tensor {module_name}: {len(suffix_matches)} matches")
    raise ValueError(
        f"could not find target weight for adapter tensor: {module_name}; "
        f"debug samples={_target_weight_debug_samples(model_state, module_name)}"
    )


def _module_name_candidates(module_name: str) -> tuple[str, ...]:
    candidates = [module_name]
    for prefix in ("base_model.model.", "model.", "language_model.", "language_model.model.", "model.language_model."):
        if module_name.startswith(prefix):
            candidates.append(module_name.removeprefix(prefix))
    if module_name.startswith("model.language_model."):
        stripped = module_name.removeprefix("model.language_model.")
        candidates.append(f"language_model.{stripped}")
        candidates.append(stripped)
    return tuple(dict.fromkeys(candidates))


def _normalized_module_name(module_name: str) -> str:
    name = module_name.removesuffix(".linear")
    changed = True
    while changed:
        changed = False
        for prefix in ("base_model.model.", "model.", "language_model.", "text_model."):
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                changed = True
    return name


def _normalized_weight_name(weight_name: str) -> str:
    return _normalized_module_name(weight_name.removesuffix(".weight").removesuffix(".linear"))


def _target_weight_debug_samples(model_state: dict[str, Any], module_name: str) -> dict[str, list[str]]:
    layer_match = re.search(r"layers\.(\d+)", module_name)
    layer_token = f"layers.{layer_match.group(1)}" if layer_match else ""
    projection = module_name.rsplit(".", maxsplit=1)[-1]
    weight_names = [name for name in model_state if name.endswith(".weight")]
    return {
        "layer": [name for name in weight_names if layer_token and layer_token in name][:12],
        "language": [name for name in weight_names if "language" in name][:12],
        "projection": [name for name in weight_names if projection in name][:12],
        "self_attn": [name for name in weight_names if "self_attn" in name][:12],
    }


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"output_dir already exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _repair_missing_base_tensors(
    *,
    base_model_id: str,
    output_dir: Path,
    hf_hub_download: Any,
    safe_open: Any,
    save_file: Any,
) -> int:
    base_files = _hub_safetensor_files(base_model_id=base_model_id, hf_hub_download=hf_hub_download, safe_open=safe_open)
    output_index_path = output_dir / "model.safetensors.index.json"
    if output_index_path.exists():
        output_index = json.loads(output_index_path.read_text(encoding="utf-8"))
        output_weight_map = dict(output_index["weight_map"])
    else:
        single_model = output_dir / "model.safetensors"
        if not single_model.exists():
            raise ValueError(f"merged output has neither model.safetensors nor index: {output_dir}")
        with safe_open(single_model, framework="pt") as model_file:
            output_weight_map = {key: single_model.name for key in model_file.keys()}
        output_index = {"metadata": {}, "weight_map": output_weight_map}

    missing_keys = sorted(set(base_files) - set(output_weight_map))
    if not missing_keys:
        return 0

    repair_tensors: dict[str, Any] = {}
    for key in missing_keys:
        with safe_open(base_files[key], framework="pt") as base_file:
            repair_tensors[key] = base_file.get_tensor(key)

    repair_file = "model-missing-base.safetensors"
    save_file(repair_tensors, output_dir / repair_file, metadata={"format": "pt"})
    for key in missing_keys:
        output_weight_map[key] = repair_file
    output_index["weight_map"] = dict(sorted(output_weight_map.items()))
    output_index.setdefault("metadata", {})
    output_index["metadata"]["total_size"] = _safetensor_total_size(output_dir, set(output_weight_map.values()), safe_open)
    output_index_path.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(missing_keys)


def _hub_safetensor_files(*, base_model_id: str, hf_hub_download: Any, safe_open: Any) -> dict[str, str]:
    try:
        index_path = Path(hf_hub_download(base_model_id, "model.safetensors.index.json", repo_type="model"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {
            key: hf_hub_download(base_model_id, filename, repo_type="model")
            for key, filename in index["weight_map"].items()
        }
    except Exception:
        model_path = hf_hub_download(base_model_id, "model.safetensors", repo_type="model")
        with safe_open(model_path, framework="pt") as model_file:
            return {key: model_path for key in model_file.keys()}


def _safetensor_total_size(output_dir: Path, filenames: set[str], safe_open: Any) -> int:
    total = 0
    for filename in filenames:
        with safe_open(output_dir / filename, framework="pt") as model_file:
            for key in model_file.keys():
                tensor = model_file.get_tensor(key)
                total += tensor.numel() * tensor.element_size()
    return total


def _write_model_card(merge_config: HfPolicyAdapterMergeConfig) -> None:
    readme = merge_config.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "library_name: transformers",
                f"base_model: {merge_config.base_model_id}",
                "tags:",
                "- gemma",
                "- telugu",
                "- medical",
                "- grpo",
                "- peft",
                "- merged",
                "---",
                "",
                f"# {merge_config.hub_model_id or merge_config.output_dir.name}",
                "",
                "Full HF/PEFT merged Gemma health model after Telugu medical GRPO policy alignment.",
                "",
                "## Merge Inputs",
                "",
                f"- Base model: `{merge_config.base_model_id}`",
                f"- Adapter: `{merge_config.adapter_model_id}`",
                f"- Merge path: manual LoRA delta add into HF model weights (`safe_merge={merge_config.safe_merge}`)",
                f"- Missing target policy: `{merge_config.missing_target_policy}`",
                f"- Dtype: `{merge_config.dtype}`",
                "",
                "Evaluate safety, Telugu quality, and medical QA behavior before clinical or field use.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _upload_large_folder(api: Any, merge_config: HfPolicyAdapterMergeConfig) -> None:
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            repo_id=merge_config.hub_model_id,
            repo_type="model",
            folder_path=str(merge_config.output_dir),
        )
        return
    api.upload_folder(
        repo_id=merge_config.hub_model_id,
        repo_type="model",
        folder_path=str(merge_config.output_dir),
    )
