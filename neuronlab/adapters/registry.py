"""Pick an adapter by config.model_type, with auto-discovery for unknown architectures."""
from __future__ import annotations

import torch
from torch import nn

from .base import DownProjAdapter, ModelAdapter
from .gated_mlp import GatedMLPAdapter
from .gpt2 import GPT2Adapter

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "qwen2": GatedMLPAdapter,
    "llama": GatedMLPAdapter,
    "mistral": GatedMLPAdapter,
    "gemma": GatedMLPAdapter,
    "gemma2": GatedMLPAdapter,
    "gpt2": GPT2Adapter,
}

DOWN_NAMES = ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2")
MLP_NAMES = ("mlp", "feed_forward", "ffn")
ATTN_NAMES = ("self_attn", "attn", "attention", "self_attention")


class AdapterError(RuntimeError):
    pass


def module_tree(model: nn.Module, max_depth: int = 4) -> str:
    lines = []
    for name, mod in model.named_modules():
        depth = name.count(".") + (1 if name else 0)
        if depth > max_depth:
            continue
        lines.append(f"{'  ' * depth}{name or '<root>'}: {type(mod).__name__}")
    return "\n".join(lines)


def register_adapter(model_type: str, cls: type[ModelAdapter]) -> None:
    """Add support for a new architecture: subclass DownProjAdapter and register it here."""
    ADAPTERS[model_type] = cls


def _find_layer_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    n_expected = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", None)
    candidates = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) > 0]
    if n_expected is not None:
        exact = [(n, m) for n, m in candidates if len(m) == n_expected]
        if exact:
            return min(exact, key=lambda nm: nm[0].count("."))
    if candidates:
        return max(candidates, key=lambda nm: len(nm[1]))
    raise AdapterError("no nn.ModuleList of decoder layers found")


def _first_child(layer: nn.Module, names) -> str | None:
    for n in names:
        if hasattr(layer, n) and isinstance(getattr(layer, n), nn.Module):
            return n
    return None


def discover_adapter(model: nn.Module) -> ModelAdapter:
    try:
        path, layers = _find_layer_list(model)
        layer0 = layers[0]
        mlp_name = _first_child(layer0, MLP_NAMES)
        attn_name = _first_child(layer0, ATTN_NAMES)
        if mlp_name is None:
            raise AdapterError(f"no MLP child named any of {MLP_NAMES} in layer 0")
        down_name = _first_child(getattr(layer0, mlp_name), DOWN_NAMES)
        if down_name is None:
            raise AdapterError(f"no down projection named any of {DOWN_NAMES} in the MLP")
        down = getattr(getattr(layer0, mlp_name), down_name)
        is_conv1d = type(down).__name__ == "Conv1D"
        base = model.base_model
        norm_path = next(
            (f"{pre}.{n}" if pre else n
             for pre in _prefixes(model, base)
             for n in ("norm", "ln_f", "final_layernorm", "final_layer_norm")
             if hasattr(_resolve(model, pre), n)),
            None,
        )
        if norm_path is None:
            raise AdapterError("could not locate the final norm")
    except (AdapterError, AttributeError) as e:
        raise AdapterError(
            f"Auto-discovery failed for {type(model).__name__}: {e}\n\nModule tree:\n{module_tree(model)}\n\n"
            "Write a DownProjAdapter subclass and call register_adapter(model_type, cls)."
        ) from e

    cls_attrs = dict(layers_path=path, mlp_name=mlp_name, attn_name=attn_name or "", down_name=down_name,
                     final_norm_path=norm_path)
    if is_conv1d:
        cls = type("DiscoveredConv1DAdapter", (GPT2Adapter,), cls_attrs)
    else:
        cls = type("DiscoveredLinearAdapter", (GatedMLPAdapter,), cls_attrs)
    return cls(model)


def _prefixes(model, base):
    names = [n for n, m in model.named_children() if m is base]
    return [""] + names


def _resolve(model, prefix):
    return getattr(model, prefix) if prefix else model


def get_adapter(model: nn.Module) -> ModelAdapter:
    model_type = getattr(model.config, "model_type", None)
    cls = ADAPTERS.get(model_type)
    if cls is not None:
        return cls(model)
    return discover_adapter(model)
