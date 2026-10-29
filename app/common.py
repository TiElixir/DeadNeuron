"""Shared UI plumbing: cached model, active run, basket, GPU lock, background jobs, styling.

The UI only calls library functions. Expensive things are cached on disk by the library and held in
memory here with st.cache_resource (one model, one experiment).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuronlab.model_loader import load_config, load_lab  # noqa: E402
from neuronlab.pipeline import describe_runs, load_experiment  # noqa: E402

import os  # noqa: E402

RUNS = Path(os.environ.get("NEURONLAB_RUNS_DIR") or ROOT / "runs")
MODE = "dark"

CSS = """
<style>
html, body, [class*="st-"] { font-family: 'SF Pro Display','Geist Sans','Helvetica Neue',system-ui,sans-serif; line-height: 1.6; }
h1, h2, h3 { font-family: 'Newsreader','Lyon Text','Iowan Old Style',Georgia,serif; font-weight: 500;
             letter-spacing: -0.02em; line-height: 1.12; }
h1 { font-size: 2.3rem; padding-top: 0.2rem; }
code, pre, kbd, [data-testid="stDataFrame"] { font-family: 'Geist Mono','SF Mono','JetBrains Mono',ui-monospace,monospace; }
[data-testid="stSidebar"] { border-right: 1px solid #2e2e2b; }
.stButton > button, .stDownloadButton > button { border-radius: 4px; box-shadow: none; border: 1px solid #3a3a37;
                                                 transition: transform 120ms ease, background-color 120ms ease; }
.stButton > button:active { transform: scale(0.98); }
.stButton > button[kind="primary"] { background: #ecebe6; color: #1a1a19; border-color: #ecebe6; }
.stButton > button[kind="primary"]:hover { background: #d8d5cb; }
[data-testid="stMetric"] { border: 1px solid #2e2e2b; border-radius: 8px; padding: 12px 16px; }
[data-testid="stMetricLabel"] { color: #8a8880; }
.muted { color: #8a8880; font-size: 0.9rem; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 9999px; font-size: 0.72rem; letter-spacing: 0.05em;
       text-transform: uppercase; }
.tag-red { background: #3a2220; color: #f0a09e; }
.tag-blue { background: #1c2f3d; color: #8ec3ea; }
.tag-green { background: #223125; color: #9cc79f; }
.tag-yellow { background: #392f17; color: #e0c27a; }
.reply { border: 1px solid #2e2e2b; border-radius: 8px; padding: 12px 16px; white-space: pre-wrap; font-size: 0.92rem; }
</style>
"""


@st.cache_resource
def _registry() -> dict:
    """Process-wide singletons: the GPU lock and the background job table."""
    return {"lock": threading.Lock(), "jobs": {}}


def gpu_lock() -> threading.Lock:
    return _registry()["lock"]


@st.cache_resource(max_entries=1, show_spinner="Loading model")
def get_lab(model_path: str, device: str, dtype: str):
    with gpu_lock():
        return load_lab(model_path, device, dtype)


@st.cache_resource(max_entries=1, show_spinner="Opening run")
def get_experiment(_lab, run_path: str):
    with gpu_lock():
        return load_experiment(_lab, run_path)


@st.cache_resource(max_entries=4, show_spinner="Computing directions")
def get_directions(_exp, run_path: str):
    from neuronlab.pipeline import ensure_directions
    with gpu_lock():
        return ensure_directions(_exp)


def get_attribution(exp, compute: bool = False):
    """Attribution arrays for the run: from memory, then disk, and only computed when asked to."""
    from neuronlab.pipeline import ensure_attribution
    if getattr(exp, "_attr", None) is not None:
        return exp._attr
    if exp.store.load_arrays("attribution") is None and not compute:
        return None
    with gpu_lock():
        exp._attr = ensure_attribution(exp)
    return exp._attr


# --- background jobs (progress, ETA, cancel) ------------------------------------------------------------------

class Job:
    def __init__(self, name):
        self.name, self.cancel_event = name, threading.Event()
        self.done, self.total, self.eta, self.status, self.error, self.result = 0, 1, 0.0, "running", None, None
        self.t0 = time.time()


def start_job(name: str, fn):
    """fn(progress, cancel) runs in a thread under the GPU lock. progress(done, total, eta_seconds)."""
    job = Job(name)

    def runner():
        try:
            with gpu_lock():
                job.result = fn(lambda d, t, eta: setattr(job, "done", d) or setattr(job, "total", max(t, 1))
                                or setattr(job, "eta", eta), job.cancel_event.is_set)
            job.status = "finished"
        except Exception as e:  # noqa: BLE001 - shown to the user
            job.status = "cancelled" if type(e).__name__ == "Cancelled" else "failed"
            job.error = None if job.status == "cancelled" else f"{type(e).__name__}: {e}"

    _registry()["jobs"][name] = job
    threading.Thread(target=runner, daemon=True).start()
    return job


def get_job(name: str) -> Job | None:
    return _registry()["jobs"].get(name)


@st.fragment(run_every=1)
def job_panel(name: str, on_finish_key: str):
    """Progress, ETA and a cancel button for a background job; polls once a second while it runs."""
    job = get_job(name)
    if job is None:
        return
    if job.status == "running":
        st.progress(min(job.done / job.total, 1.0), text=f"{job.name}: {job.done}/{job.total}, about {job.eta:.0f} s left")
        if st.button("Cancel", key=f"cancel_{name}"):
            job.cancel_event.set()
    elif job.status == "finished":
        st.session_state[on_finish_key] = job.result
        st.success(f"{job.name} finished in {time.time() - job.t0:.0f} s.")
    elif job.status == "cancelled":
        st.warning(f"{job.name} was cancelled. Partial results were not kept; start it again to rerun.")
    else:
        st.error(f"{job.name} failed: {job.error}")


# --- page chrome ------------------------------------------------------------------------------------------------

def gpu_text() -> str:
    if not torch.cuda.is_available():
        return "no CUDA device"
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.2f} / {total:.1f} GB allocated"


def sidebar():
    cfg = load_config()
    ss = st.session_state
    ss.setdefault("basket", [])
    ss.setdefault("active_run", None)
    ss.setdefault("sel_layer", 0)
    ss.setdefault("sel_neuron", 0)
    with st.sidebar:
        st.markdown("### NeuronLab")
        path = st.text_input("Model directory", cfg.get("model_path", ""), key="model_path",
                             help="A local directory with config.json and weights. NeuronLab never downloads models.")
        c1, c2 = st.columns(2)
        device = c1.selectbox("Device", ["auto", "cuda", "cpu"], key="device")
        dtype = c2.selectbox("Dtype", ["bfloat16", "float16", "float32"], key="dtype")
        lab = None
        p = Path(path) if Path(path).is_absolute() else ROOT / path
        if not (p / "config.json").exists():
            st.error(f"No model at {p}. Set the directory above to a local model folder (it needs config.json).")
        else:
            try:
                lab = get_lab(str(p), device, dtype)
                st.caption(f"Loaded {lab.adapter.describe()['model_type']}, {lab.adapter.n_layers} layers on "
                           f"{lab.device.type} ({str(lab.dtype).replace('torch.', '')})")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load the model: {e}")
        st.caption(f"GPU memory: {gpu_text()}")
        st.divider()
        runs = [r for r in describe_runs(RUNS) if lab is None or r["model_fp"] == lab.fingerprint]
        labels = {r["path"]: f"{r['name']}  ·  {r['created'][:16]}  ·  {r['path'][-10:]}" for r in runs}
        options = [None] + list(labels)
        cur = ss.active_run if ss.active_run in labels else None
        pick = st.selectbox("Active run", options, index=options.index(cur), key="run_pick",
                            format_func=lambda x: "None selected" if x is None else labels[x],
                            help="Saved runs for this model. Create a new one under Experiment setup.")
        ss.active_run = pick
        st.caption(f"Basket: {len(ss.basket)} neurons")
    return lab


def page(title: str, subtitle: str | None = None):
    st.set_page_config(page_title=f"{title} · NeuronLab", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    lab = sidebar()
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='muted'>{subtitle}</p>", unsafe_allow_html=True)
    return lab


def require_experiment(lab):
    """Return the active Experiment or stop the page with a message that names the next action."""
    if lab is None:
        st.info("Load a model in the sidebar first.")
        st.stop()
    if not st.session_state.active_run:
        st.info("No run is selected. Pick a saved run in the sidebar, or create one under Experiment setup.")
        st.page_link("pages/2_Experiment_setup.py", label="Go to Experiment setup")
        st.stop()
    try:
        return get_experiment(lab, st.session_state.active_run)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not open this run: {e}")
        st.stop()


def basket_add(pairs):
    b = st.session_state.basket
    seen = set(map(tuple, b))
    for p in pairs:
        p = (int(p[0]), int(p[1]))
        if p not in seen:
            b.append(p)
            seen.add(p)
