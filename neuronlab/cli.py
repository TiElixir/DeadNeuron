"""Command line front end. Calls the same library functions as the examples and the UI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .interventions import Mode
from .model_loader import ROOT, load_lab
from .pipeline import (Settings, format_comparison, format_top_neurons, prepare, run_ablation)


def _mode(s: str) -> Mode:
    name, _, arg = s.partition(":")
    if name in ("zero", "mean"):
        return getattr(Mode, name)()
    if name in ("clamp", "scale"):
        return getattr(Mode, name)(float(arg))
    if name == "push":
        alpha, _, sign = arg.partition(",")
        return Mode.push(float(alpha), -1 if sign.strip() in ("-", "-1") else 1)
    raise argparse.ArgumentTypeError(f"unknown mode {s!r} (zero | mean | clamp:V | scale:F | push:ALPHA,SIGN)")


def _preset_path(p: str) -> Path:
    path = Path(p)
    return path if path.exists() else ROOT / "presets" / (p if p.endswith(".yaml") else p + ".yaml")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="neuronlab", description=__doc__)
    ap.add_argument("--model", help="local model directory (default: config.yaml / NEURONLAB_MODEL_PATH)")
    ap.add_argument("--device"), ap.add_argument("--dtype")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("describe", help="print the model structure and confirm the neuron site")

    def common(p):
        p.add_argument("preset"), p.add_argument("--runs-dir")
        p.add_argument("--split-seed", type=int, default=0), p.add_argument("--n-per-side", type=int, default=60)
        p.add_argument("--mask", default="user_content"), p.add_argument("--reduction", default="mean")
        p.add_argument("--n-perm", type=int, default=500)
        p.add_argument("--top", type=int, default=20)
        p.add_argument("--sign", default="abs", choices=["abs", "pos", "neg"])

    common(sub.add_parser("score", help="record the training split and print the top neurons"))
    ab = sub.add_parser("ablate", help="ablate the top-k neurons and compare with a random baseline")
    common(ab)
    ab.add_argument("-k", type=int, default=50), ab.add_argument("--mode", type=_mode, default=Mode.mean())
    ab.add_argument("--positions", default="all", choices=["all", "prompt_only", "generated_only"])
    ab.add_argument("--n-random", type=int, default=5), ab.add_argument("--json", help="write results to this file")
    args = ap.parse_args(argv)

    lab = load_lab(args.model, args.device, args.dtype)
    if args.cmd == "describe":
        d = lab.adapter.describe()
        print({k: v for k, v in d.items() if k != "layers"})
        print("neuron site (layer 0):", type(lab.adapter.neuron_site(0)).__name__, "-> input = neuron activations")
        return 0

    st = Settings(args.split_seed, args.n_per_side, args.mask, args.reduction, args.n_perm)
    exp = prepare(lab, _preset_path(args.preset), st, args.runs_dir,
                  progress=lambda d, t, eta: print(f"\r  recording {d}/{t}  eta {eta:5.0f}s", end="", file=sys.stderr))
    print(file=sys.stderr)
    r = exp.contrast
    print(f"n_pos={r.n_pos} n_neg={r.n_neg} n_control={r.n_control} (training split, seed {st.split_seed}); "
          f"{int(r.significant.sum())} significant, {int((r.significant & ~r.generic).sum())} after hiding generic")
    top = exp.top_neurons(max(args.top, getattr(args, "k", 0)), args.sign)
    print(format_top_neurons(exp, top[:args.top]))
    if args.cmd == "ablate":
        chosen = top[:args.k]
        if len(chosen) < args.k:
            print(f"NOTE: only {len(chosen)} significant, non-generic candidates; ablating those.")
        cmp_ = run_ablation(exp, chosen, args.mode, args.positions, args.n_random)
        print(f"\nAblating {len(chosen)} neurons, mode={args.mode}, positions={args.positions}, ranking sign={args.sign}")
        print(format_comparison(cmp_, {}))
        if args.json:
            Path(args.json).write_text(json.dumps({"neurons": chosen, "stats": cmp_.stats,
                                                   "per_probe": cmp_.real.per_probe}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
