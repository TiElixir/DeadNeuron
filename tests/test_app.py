"""Headless run of every Streamlit page against the real model (skipped without NEURONLAB_MODEL_PATH).

AppTest executes the page scripts and simulates widgets, so it catches exceptions and wrong wiring. It cannot
click inside a Plotly chart: chart selection is covered by the plain selectors that sit next to each chart.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("NEURONLAB_MODEL_PATH"), reason="NEURONLAB_MODEL_PATH not set")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["app/Home.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "app" / "pages").glob("*.py"))


@pytest.fixture(scope="module")
def real_lab():
    """One model for the whole module: the app's get_lab is pointed at it so no second copy is loaded."""
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    import common
    from neuronlab.model_loader import load_lab
    lab = load_lab()
    original = common.get_lab
    common.get_lab = lambda *a, **k: lab
    yield lab
    common.get_lab = original


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, real_lab):
    runs = tmp_path_factory.mktemp("runs")
    os.environ["NEURONLAB_RUNS_DIR"] = str(runs)
    import importlib, common
    importlib.reload(common) if False else setattr(common, "RUNS", runs)
    from neuronlab.pipeline import Settings, prepare
    lab = real_lab
    exp = prepare(lab, ROOT / "presets" / "aeroplane_vs_submarine.yaml", Settings(n_per_side=14, n_perm=50), runs)
    yield str(exp.store.dir)
    os.environ.pop("NEURONLAB_RUNS_DIR", None)


def run_page(page, run_dir, state=None):
    """Enter through Home.py and switch to the page, as the real multipage app does."""
    at = AppTest.from_file(str(ROOT / "app/Home.py"), default_timeout=180)
    at.session_state["active_run"] = run_dir
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    if page != "app/Home.py":
        at.switch_page(page.replace("app/", "", 1))
        at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page, run_dir, real_lab):
    at = run_page(page, run_dir)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_pages_without_a_run_explain_the_next_step(run_dir, real_lab):
    at = run_page("app/pages/3_Layer_overview.py", run_dir, {"active_run": None})
    assert not at.exception, [e.value for e in at.exception]
    assert any("No run is selected" in i.value for i in at.info)


def test_basket_flow_and_evaluation_is_cached_on_disk(run_dir, real_lab):
    at = run_page("app/pages/6_Basket_and_ablation_lab.py", run_dir, {"basket": [(5, 10), (6, 20), (7, 30)]})
    assert not at.exception
    assert any(b.label == "Run evaluation" for b in at.button)
    from neuronlab.interventions import Mode
    from neuronlab.pipeline import evaluate_basket, first_heldout_prompts, load_experiment
    exp = load_experiment(real_lab, run_dir)
    gen = first_heldout_prompts(exp, 3)
    res = evaluate_basket(exp, [(5, 10), (6, 20), (7, 30)], Mode.mean(), "all", 5, ("train", "heldout"), gen, False, None, 120)
    assert not res["from_cache"]
    again = evaluate_basket(exp, [(5, 10), (6, 20), (7, 30)], Mode.mean(), "all", 5, ("train", "heldout"), gen, False, None, 120)
    assert again["from_cache"] and again["stats"]["d_pos"]["real"] == res["stats"]["d_pos"]["real"]
    # the page now shows the stored result without recomputation
    at = run_page("app/pages/6_Basket_and_ablation_lab.py", run_dir, {"basket": [(5, 10), (6, 20), (7, 30)]})
    assert not at.exception, [e.value for e in at.exception]
    assert any("Loaded from disk" in c.value for c in at.caption)


def test_neuron_card_and_drilldown_with_a_selected_neuron(run_dir, real_lab):
    st_ = {"sel_layer": 12, "sel_neuron": 100}
    for page in ("app/pages/5_Neuron_card.py", "app/pages/4_Layer_drill-down.py"):
        at = run_page(page, run_dir, st_)
        assert not at.exception, (page, [e.value for e in at.exception])
