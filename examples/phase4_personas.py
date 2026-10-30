"""Phase 4 driver.

    python examples/phase4_personas.py search [persona name]       # target-persona search (default: Ford Mustang)
    python examples/phase4_personas.py sweep [neurons_per_layer]   # persona sweep; resumable (rerun to resume)
    python examples/phase4_personas.py report                      # labels + the 10 most interesting personas
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronlab.model_loader import ROOT, load_lab
from neuronlab.personas import (SweepConfig, persona_search_store, sweep_store, estimate_eta, interesting_personas, load_labels, load_personas, persona_search,
                                run_sweep, sweep_size, sweep_table)
from neuronlab.pipeline import ensure_calibration
from neuronlab.storage import RunStore

import os
RULE = os.environ.get("LAYER_RULE", "auroc_then_margin")
cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
lab = load_lab()
ensure_calibration(lab)
tick = lambda d, t, eta: print(f"\r  {d}/{t}  eta {eta:6.0f}s", end="", flush=True)

if cmd == "search":
    name = sys.argv[2] if len(sys.argv) > 2 else "Ford Mustang"
    persona = next(p for p in load_personas(ROOT / "presets" / "personas.yaml") if p.name.lower() == name.lower())
    store = persona_search_store(lab, persona, RULE)
    t0 = time.time()
    res = store.load_json("search")
    if res is None:
        res = persona_search(lab, persona, layer_rule=RULE, progress=tick)
        store.save_json("search", res)
    print(f"\n{persona.name}: {time.time() - t0:.0f}s; train questions {res['n']['train']}, held-out {res['n']['heldout']}; "
          f"layer rule {res.get('layer_rule', 'auroc')}; best layers {res['best_layers']} (held-out AUROC {[round(res['separation'][l], 2) for l in res['best_layers']]})")
    print(f"success = reply has a persona keyword AND does not claim to be an AI AND is coherent (rep<=0.30, ppl<=40); n={res['n']['heldout']} held-out questions per row")
    print(f"{'method':<20}{'strength':<12}{'success':>8}{'(random)':>9}{'keyword':>8}{'AI-claim':>9}{'coherent':>9}{'1st-pers':>9}{'rep':>6}{'ppl':>7}  detail")
    for r in sorted(res["rows"], key=lambda r: (-r["success_rate"], -r["keyword_rate"])):
        rs = "" if r["random_success"] is None else f"{r['random_success']:.2f}"
        print(f"{r['method']:<20}{r['strength']:<12}{r['success_rate']:>8.2f}{rs:>9}{r['keyword_rate']:>8.2f}{r['ai_claim_rate']:>9.2f}"
              f"{r['coherent_rate']:>9.2f}{r.get('first_person_rate', float('nan')):>9.2f}{r['rep_mean']:>6.2f}{(r['ppl_median'] or 0):>7.1f}  {r['detail']}")
elif cmd in ("sweep", "report"):
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    cfg = SweepConfig(layers=list(range(lab.adapter.n_layers)), neurons_per_layer=per)
    store = sweep_store(lab, cfg)
    if cmd == "sweep":
        sz = sweep_size(lab, cfg)
        eta = estimate_eta(lab, cfg, sz["items"])
        print(f"sweep: {sz['items']:,} replies at the first rung (up to {sz['max_generations']:,} with retries); full sweep would be "
              f"{sz['full_sweep_items']:,}. Measured {eta['replies_per_s']:.1f} replies/s -> first pass about {eta['first_pass_s'] / 60:.0f} min")
        t0 = time.time()
        run_sweep(lab, store, cfg, progress=tick)
        print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")
    rows = sweep_table(store.load_json("sweep_state")["records"], load_labels(store))
    from collections import Counter
    print("categories:", dict(Counter(r["category"] for r in rows)), "| coherent:", sum(r["coherent"] for r in rows), "of", len(rows))
    print("alpha used:", dict(Counter(r["alpha"] for r in rows)))
    print("\n10 most interesting personas (persona label, coherent, distinct name; strongest alpha, then fluency):")
    for i, r in enumerate(interesting_personas(rows, 10), 1):
        print(f"{i:>2}. layer {r['layer']:>2} neuron {r['neuron']:>4} {r['direction']:<4} alpha {r['alpha']}  -> {r['persona_name']!r}\n      {r['reply'][:170]!r}")
