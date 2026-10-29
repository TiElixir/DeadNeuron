"""Load a model + tokenizer from a local path. Never touches the network."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import ModelAdapter, get_adapter

ROOT = Path(__file__).resolve().parent.parent
DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
          "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = dict(cfg or {})
    env = os.environ.get("NEURONLAB_MODEL_PATH")
    if env:
        cfg["model_path"] = env
    return cfg


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def model_fingerprint(model_path: str | Path) -> str:
    """Stable id for a model directory: resolved path + config.json + weight file sizes."""
    p = Path(model_path).resolve()
    h = hashlib.sha1(str(p).encode())
    cfg = p / "config.json"
    if cfg.exists():
        h.update(cfg.read_bytes())
    for f in sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    return h.hexdigest()[:12]


@dataclass
class Lab:
    """Everything an experiment needs: model, tokenizer, adapter and caches."""

    model: torch.nn.Module
    tokenizer: object
    adapter: ModelAdapter
    device: torch.device
    dtype: torch.dtype
    model_path: str = "<in-memory>"
    fingerprint: str = "memory"
    batch_sizes: dict = field(default_factory=lambda: {"record": 8, "eval": 8, "generate": 8, "attribution": 2})
    cache: dict = field(default_factory=dict)  # r_L, control means, normalised out-dir norms...

    def intervene(self, spec, boundary=None):
        from .interventions import intervene

        return intervene(self, spec, boundary=boundary)

    @property
    def n_layers(self) -> int:
        return self.adapter.n_layers

    def out_dirs(self, layer: int) -> torch.Tensor:
        key = ("out_dirs", layer)
        if key not in self.cache:
            self.cache[key] = self.adapter.neuron_out_dirs(layer).float().to(self.device)
        return self.cache[key]


def build_lab(model, tokenizer, device=None, model_path="<in-memory>", fingerprint="memory",
              batch_sizes=None) -> Lab:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device(device) if device else next(model.parameters()).device
    lab = Lab(model=model, tokenizer=tokenizer, adapter=get_adapter(model), device=device,
              dtype=next(model.parameters()).dtype, model_path=str(model_path), fingerprint=fingerprint)
    if batch_sizes:
        lab.batch_sizes.update(batch_sizes)
    return lab


def load_lab(model_path: str | Path | None = None, device: str | None = None, dtype: str | None = None,
             config_path: str | Path | None = None) -> Lab:
    cfg = load_config(config_path)
    model_path = model_path or cfg.get("model_path")
    if not model_path or "<PATH" in str(model_path):
        raise FileNotFoundError("No model path configured. Set model_path in config.yaml or NEURONLAB_MODEL_PATH.")
    p = Path(model_path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"{p} is not a model directory (no config.json). NeuronLab never downloads models.")
    dev = resolve_device(device or cfg.get("device", "auto"))
    dt = DTYPES[(dtype or cfg.get("dtype", "bfloat16"))]
    if dev.type == "cpu" and dt == torch.float16:
        dt = torch.float32
    tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(p, local_files_only=True, dtype=dt).to(dev)
    return build_lab(model, tok, device=dev, model_path=p, fingerprint=model_fingerprint(p),
                     batch_sizes=cfg.get("batch_sizes"))
