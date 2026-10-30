import html
import re

import pandas as pd
import streamlit as st

from common import ROOT, job_panel, page, start_job
from neuronlab.personas import PersonaSpec, load_personas, persona_search, persona_search_store
from neuronlab.pipeline import ensure_calibration

lab = page("Persona search", "Find a direction or neurons that make the default assistant answer as a chosen persona.")
if lab is None:
    st.info("Load a model in the sidebar first.")
    st.stop()
ss = st.session_state

personas = load_personas(ROOT / "presets" / "personas.yaml")
questions = personas[0].questions
names = [p.name for p in personas] + ["Custom persona"]
choice = st.selectbox("Persona", names)
if choice == "Custom persona":
    name = st.text_input("Name", placeholder="e.g. Lighthouse keeper")
    prompt = st.text_area("System prompt", placeholder="You are a lighthouse keeper. Always answer in the first person as the keeper, never as an AI.")
    kws = st.text_input("Keywords (comma separated, 8 to 12)", placeholder="lighthouse, lamp, storm, sea, ...")
    qtxt = st.text_area("Elicitation questions (one per line)", "\n".join(questions), height=160)
    persona = PersonaSpec(name.strip() or "custom", prompt.strip(), [k.strip() for k in kws.split(",") if k.strip()],
                          [q.strip() for q in qtxt.splitlines() if q.strip()])
    ready = bool(name.strip() and prompt.strip() and len(persona.keywords) >= 3 and len(persona.questions) >= 10)
    if not ready:
        st.info("Fill in the name, system prompt, at least three keywords and at least ten questions to run the search.")
else:
    persona = next(p for p in personas if p.name == choice)
    ready = True
    with st.expander("Persona definition"):
        st.write(persona.system_prompt)
        st.write("Keywords: " + ", ".join(persona.keywords))

st.caption("Method: answer the questions with and without the persona system prompt, find the residual direction and the "
           "neurons that separate the two, then try to reproduce the persona on held-out questions **without** the system "
           "prompt. A reply counts as a success if it contains a persona keyword, does not claim to be an AI, and is coherent "
           "(at most 30% repeated 4-grams, reply perplexity at most 40). Every row has a random control.")
n_random = int(st.number_input("Random controls per row", 1, 6, 3))
max_new = int(st.number_input("Max new tokens", 40, 200, 100))

if ready:
    store = persona_search_store(lab, persona)
    res = store.load_json("search")
    if res is None:
        st.write("No saved search for this persona yet. It takes a few minutes on a small GPU.")
        if st.button("Run persona search", type="primary"):
            def job(progress, cancel):
                ensure_calibration(lab)
                out = persona_search(lab, persona, n_random=n_random, max_new=max_new, progress=progress, cancel=cancel)
                store.save_json("search", out)
                return True
            start_job("Persona search", job)
        job_panel("Persona search", "persona_done")
        if ss.pop("persona_done", None):
            st.rerun()
    else:
        st.caption("Loaded from disk.")
        n = res["n"]
        st.markdown(f"**{res['persona']}**: {n['train']} training questions, **{n['heldout']} held-out questions per row**, so each "
                    f"rate moves in steps of {1 / n['heldout']:.1f}. Direction layers (by held-out AUROC, ties broken by margin): "
                    f"{res['best_layers']}.")
        df = pd.DataFrame(res["rows"])
        df["config"] = df["method"] + " | " + df["strength"] + " | " + df["detail"]
        view = df.sort_values(["success_rate", "keyword_rate"], ascending=False)[
            ["method", "strength", "detail", "success_rate", "random_success", "keyword_rate", "ai_claim_rate", "coherent_rate",
             "first_person_rate", "rep_mean", "ppl_median"]]
        st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
            "success_rate": st.column_config.NumberColumn("success", format="%.2f"),
            "random_success": st.column_config.NumberColumn("random control", format="%.2f"),
            "keyword_rate": st.column_config.NumberColumn("keyword", format="%.2f"),
            "ai_claim_rate": st.column_config.NumberColumn("AI claim", format="%.2f"),
            "coherent_rate": st.column_config.NumberColumn("coherent", format="%.2f"),
            "first_person_rate": st.column_config.NumberColumn("first person", format="%.2f"),
            "rep_mean": st.column_config.NumberColumn("repeat", format="%.2f"),
            "ppl_median": st.column_config.NumberColumn("reply ppl", format="%.1f")})
        st.caption("Success is the stated metric. 'first person' is a diagnostic only: it separates a persona speaking as itself "
                   "from a reply that merely talks about the topic. Read the examples before trusting any row.")
        pick = st.selectbox("Show example replies for", view.index.map(lambda i: df.loc[i, "config"]).tolist())
        key = f"{df[df['config'] == pick].iloc[0]['method']}|{df[df['config'] == pick].iloc[0]['strength']}|{df[df['config'] == pick].iloc[0]['detail']}"
        for e in res["examples"].get(key, []):
            st.markdown(f"**{e['question']}**<div class='reply'>{html.escape(e['reply'])}</div>", unsafe_allow_html=True)
        if st.button("Delete this saved search and rerun"):
            (store.dir / "search.json").unlink()
            st.rerun()
