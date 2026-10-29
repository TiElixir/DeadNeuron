"""Run folders, atomic saves and resumable checkpoints.

Layout: <runs_dir>/<model fingerprint>/<preset hash>/<settings hash>/<name>.{npz,json,npy}
Anything expensive is looked up here before it is recomputed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .recording import SetRecording


def settings_hash(settings: dict) -> str:
    return hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _atomic(path: Path, write):
    """write(fileobj) into a temp file, then rename over `path`; a failed write leaves `path` untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class RunStore:
    def __init__(self, root: str | Path, model_fp: str, preset_hash: str, settings: dict | None = None):
        self.settings = settings or {}
        self.dir = Path(root) / model_fp / preset_hash / settings_hash(self.settings)
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (self.dir / "run.json").exists():
            self.save_json("run", {"model_fp": model_fp, "preset_hash": preset_hash, "settings": self.settings,
                                   "created": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform()})

    # arrays + meta -----------------------------------------------------------
    def has(self, name: str) -> bool:
        return any((self.dir / f"{name}{ext}").exists() for ext in (".npz", ".json", ".npy"))

    def save_arrays(self, name: str, arrays: dict, meta: dict | None = None) -> None:
        payload = dict(arrays)
        payload["__meta__"] = np.frombuffer(json.dumps(meta or {}, default=str).encode(), dtype=np.uint8)
        _atomic(self.dir / f"{name}.npz", lambda f: np.savez(f, **payload))

    def load_arrays(self, name: str):
        path = self.dir / f"{name}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "__meta__"}
            meta = json.loads(bytes(z["__meta__"]).decode()) if "__meta__" in z.files else {}
        return arrays, meta

    def save_json(self, name: str, obj) -> None:
        _atomic(self.dir / f"{name}.json", lambda f: f.write(json.dumps(obj, indent=2, default=str).encode()))

    def load_json(self, name: str):
        path = self.dir / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    # recordings: per-prompt matrix as float16 .npy, everything else float32 .npz --------------------
    def save_recording(self, name: str, rec: SetRecording) -> None:
        _atomic(self.dir / f"{name}.acts.npy", lambda f: np.save(f, rec.acts))
        arrays = dict(valid=rec.valid, mean=rec.mean, var=rec.var, frac_pos=rec.frac_pos)
        if rec.resid is not None:
            arrays["resid"] = rec.resid
        self.save_arrays(f"{name}.stats", arrays, dict(meta=rec.meta, texts=rec.texts))

    def load_recording(self, name: str) -> SetRecording | None:
        got = self.load_arrays(f"{name}.stats")
        acts = self.dir / f"{name}.acts.npy"
        if got is None or not acts.exists():
            return None
        a, m = got
        return SetRecording(texts=m["texts"], acts=np.load(acts), resid=a.get("resid"), valid=a["valid"],
                            mean=a["mean"], var=a["var"], frac_pos=a["frac_pos"], meta=m["meta"])


class Checkpoint:
    """Resumable long jobs: save(state) every N batches, load() after a crash."""

    def __init__(self, store: RunStore, name: str):
        self.store, self.name = store, name

    def save(self, arrays: dict, meta: dict) -> None:
        self.store.save_arrays(f"ckpt_{self.name}", arrays, meta)

    def load(self):
        return self.store.load_arrays(f"ckpt_{self.name}")

    def clear(self) -> None:
        p = self.store.dir / f"ckpt_{self.name}.npz"
        if p.exists():
            p.unlink()


def list_runs(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/*/run.json"))
