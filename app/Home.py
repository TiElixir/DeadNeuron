"""NeuronLab entry page."""
import streamlit as st

from common import RUNS, page
from neuronlab.pipeline import describe_runs

lab = page("NeuronLab", "Find the MLP neurons that separate a concept from its contrast, then test whether they cause it.")

if lab is None:
    st.info("Point the sidebar at a local model directory to begin. NeuronLab only loads models from disk.")
    st.stop()

c1, c2, c3 = st.columns(3)
d = lab.adapter.describe()
c1.metric("Layers", d["n_layers"])
c2.metric("MLP neurons", f"{d['n_neurons']:,}")
c3.metric("Residual width", d["d_model"])

st.subheader("Saved runs")
runs = [r for r in describe_runs(RUNS) if r["model_fp"] == lab.fingerprint]
if not runs:
    st.write("There are no runs for this model yet. Start one under Experiment setup: choose a preset, then Run.")
    st.page_link("pages/2_Experiment_setup.py", label="Open Experiment setup")
else:
    st.dataframe([{"preset": r["name"], "created": r["created"], "prompts per side": r["n_per_side"],
                   "dtype": r["dtype"], "id": r["path"][-10:]} for r in runs], hide_index=True, width="stretch")
    st.caption("Pick one in the sidebar to open it. Reopening reads everything from disk; nothing is recomputed.")

st.subheader("Workflow")
st.markdown(
    "1. **Experiment setup**: choose or write positive and negative prompts, then run the recording.\n"
    "2. **Layer overview** and **Layer drill-down**: see where the contrast lives, layer by layer and neuron by neuron.\n"
    "3. **Neuron card**: inspect one neuron, push it up or down, add it to the basket.\n"
    "4. **Basket and ablation lab**: ablate the selection and compare it with random neurons.\n"
    "5. **Directions and steering**: the residual-stream alternative.")
