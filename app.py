"""BantayLigtas - Streamlit app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from bantayligtas import __version__, config
from bantayligtas.classifiers import ClassificationService, ModelUnavailableError
from bantayligtas.database import delete_report, fetch_reports, init_db, save_report

st.set_page_config(page_title="BantayLigtas", page_icon="🛟", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#d9480f"

# Accent colour per category: used for the result card border and the dashboard chart
CATEGORY_COLOURS = {
    "requests_or_urgent_needs": "#d9480f",
    "injured_or_dead_people": "#e03131",
    "rescue_volunteering_or_donation_effort": "#2f9e44",
    "displaced_people_and_evacuations": "#7048e8",
    "missing_or_found_people": "#1c7ed6",
    "infrastructure_and_utility_damage": "#868e96",
    "caution_and_advice": "#f08c00",
    "other_relevant_information": "#0c8599",
    "sympathy_and_support": "#d6336c",
    "not_humanitarian": "#adb5bd",
}

st.markdown("""
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1080px;}

/* Header */
.bl-hero {border-radius: 18px; padding: 1.5rem 1.75rem 1.35rem; margin: 0.25rem 0 0.75rem;
          color: #fff; background: linear-gradient(130deg, #c2410c 0%, #d9480f 45%, #f59f00 100%);
          box-shadow: 0 8px 24px rgba(217, 72, 15, 0.25);}
.bl-hero-top {display: flex; align-items: center; gap: 0.6rem;}
.bl-logo {font-size: 2rem; line-height: 1;}
.bl-hero h1 {color: #fff; font-size: 1.85rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -0.01em;}
.bl-hero p {margin: 0.45rem 0 0; font-size: 1.02rem; opacity: 0.95; max-width: 46rem;}
.bl-tags {margin-top: 0.8rem;}
.bl-tag {display: inline-block; padding: 0.15rem 0.65rem; margin: 0 0.35rem 0.35rem 0; border-radius: 999px;
         font-size: 0.78rem; background: rgba(255, 255, 255, 0.18); border: 1px solid rgba(255, 255, 255, 0.3);}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 0.25rem;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600; padding: 0.55rem 0.9rem;}

/* "How it works" steps */
.bl-steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin: 0.2rem 0 1rem;}
.bl-step {border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 12px; padding: 0.65rem 0.8rem;
          font-size: 0.85rem; line-height: 1.35;}
.bl-step-num {display: inline-flex; align-items: center; justify-content: center; width: 1.45rem; height: 1.45rem;
              border-radius: 50%; background: rgba(217, 72, 15, 0.12); color: #d9480f; font-weight: 700;
              font-size: 0.8rem; margin-right: 0.35rem;}
.bl-step b {font-size: 0.92rem;}
.bl-step span.bl-muted {display: block; opacity: 0.7; margin-top: 0.2rem;}

/* Result card */
.st-key-result_card {border-left: 6px solid var(--bl-accent, #d9480f) !important;}
.bl-category {font-size: 1.65rem; font-weight: 800; line-height: 1.2; margin: 0.15rem 0 0.2rem;}
.bl-kicker {font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6;}
.bl-chip {display: inline-block; padding: 0.08rem 0.55rem; margin: 0 0.25rem 0.3rem 0;
          border-radius: 999px; font-size: 0.8rem; background: rgba(217, 72, 15, 0.1);}

/* Section headings */
h3.bl-section {font-size: 1.15rem; font-weight: 700; margin: 0.25rem 0 0.5rem; padding: 0;}

/* Footer */
.bl-footer {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(128, 128, 128, 0.2);
            font-size: 0.82rem; opacity: 0.75; text-align: center;}

@media (max-width: 640px) {
  .bl-hero {padding: 1.15rem 1.15rem 1rem; border-radius: 14px;}
  .bl-hero h1 {font-size: 1.45rem;}
  .bl-hero p {font-size: 0.93rem;}
  .block-container {padding-top: 3.25rem;}
  .bl-steps {gap: 0.35rem;}
  .bl-step {padding: 0.45rem 0.4rem; text-align: center;}
  .bl-step-num {display: flex; margin: 0 auto 0.2rem;}
  .bl-step span.bl-muted {display: none;}
  .stTabs [data-baseweb="tab"] {font-size: 0.92rem; padding: 0.5rem 0.55rem;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service() -> ClassificationService:
    return ClassificationService()


init_db()
service = get_service()
status = service.status()

EXAMPLES = {
    "Urgent need (EN)": "Please send food and drinking water to the evacuation center, kids have not eaten since yesterday.",
    "Damage (EN)": "The main bridge collapsed after the earthquake and roads to the town are closed.",
    "Rescue (Taglish)": "May pamilya na na-trap sa second floor. Need rescue po.",
    "Missing person (Taglish)": "Hindi pa rin mahanap ang kapatid ko after ng baha.",
    "Caution (Filipino)": "Iwasan ang pagdaan sa kalsadang binaha.",
}

CATEGORY_ICONS = {
    "caution_and_advice": "⚠️",
    "displaced_people_and_evacuations": "🏕️",
    "infrastructure_and_utility_damage": "🏚️",
    "injured_or_dead_people": "🚑",
    "missing_or_found_people": "🔎",
    "not_humanitarian": "💬",
    "other_relevant_information": "ℹ️",
    "requests_or_urgent_needs": "🆘",
    "rescue_volunteering_or_donation_effort": "🤝",
    "sympathy_and_support": "💙",
}

LANGUAGE_NAMES = {"english": "English", "taglish": "Taglish"}
MODES = {"Auto-detect": "auto", "English": "english", "Taglish / Filipino": "taglish"}


def confidence_level(confidence: float) -> tuple[str, str]:
    """Return a (label, colour) pair for a confidence value."""
    if confidence >= 0.75:
        return "High", "green"
    if confidence >= config.LOW_CONFIDENCE_THRESHOLD:
        return "Medium", "orange"
    return "Low", "red"


def use_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state.report_text = EXAMPLES[choice]


def clear_form() -> None:
    st.session_state.report_text = ""
    st.session_state.example = None
    st.session_state.pop("last", None)


def section(title: str) -> None:
    st.markdown(f'<h3 class="bl-section">{title}</h3>', unsafe_allow_html=True)


def bar_chart(series: pd.Series, label_limit: int = 220, colours: dict[str, str] | None = None) -> alt.Chart:
    data = series.rename_axis("label").reset_index(name="count")
    colour = (alt.Color("label:N", legend=None,
                        scale=alt.Scale(domain=list(colours), range=list(colours.values())))
              if colours else alt.value(PRIMARY))
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            x=alt.X("count:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d", grid=True)),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=label_limit)),
            color=colour,
            tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("count:Q", title="Reports")],
        )
        .properties(height=max(34 * len(data), 90))
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛟 BantayLigtas")
    st.caption(f"Disaster report classifier · v{__version__}")

    st.subheader("Model status")
    st.markdown(("✅" if status["english_files"] else "❌") + " English (TF-IDF)")
    if status["taglish_files"] and status["taglish_libraries"]:
        st.markdown("✅ Taglish (DOST-ASTI RoBERTa)")
    elif status["taglish_files"]:
        st.markdown("⚠️ Taglish: install torch + transformers")
    else:
        st.markdown("❌ Taglish: falls back to English")
    with st.expander("Model files"):
        st.caption(f"Models folder:\n\n`{status['model_dir']}`")

    st.divider()
    st.caption("Research prototype. It does not contact or dispatch emergency services.")
    st.markdown("**In an emergency call 911.**")

# ---------------------------------------------------------------- header
st.markdown("""
<div class="bl-hero">
  <div class="bl-hero-top"><span class="bl-logo">🛟</span><h1>BantayLigtas</h1></div>
  <p>From scattered citizen reports to structured disaster information.</p>
  <div class="bl-tags">
    <span class="bl-tag">English</span><span class="bl-tag">Filipino</span><span class="bl-tag">Taglish</span>
    <span class="bl-tag">10 humanitarian categories</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
submit_tab, dashboard_tab, about_tab = st.tabs(["📝 Submit report", "📊 Dashboard", "ℹ️ About"])

with submit_tab:
    st.markdown("""
<div class="bl-steps">
  <div class="bl-step"><span class="bl-step-num">1</span><b>Describe</b>
    <span class="bl-muted">what is happening, in your own words.</span></div>
  <div class="bl-step"><span class="bl-step-num">2</span><b>Classify</b>
    <span class="bl-muted">BantayLigtas finds the category.</span></div>
  <div class="bl-step"><span class="bl-step-num">3</span><b>Visualize</b>
    <span class="bl-muted">reports add up on the dashboard.</span></div>
</div>
""", unsafe_allow_html=True)

    if not status["english_files"]:
        st.error("The English model files are missing, so reports can't be classified yet. "
                 f"Copy `{config.ENGLISH_VECTORIZER_FILE}` and `{config.ENGLISH_CLASSIFIER_FILE}` "
                 f"into `{status['model_dir']}` (see README), or run `python scripts/train_english.py`.")

    input_col = st.container()

    with input_col:
        with st.form("report_form", border=True):
            section("What is happening?")
            text = st.text_area("Report", key="report_text", height=140, label_visibility="collapsed",
                                placeholder="e.g. Baha na sa amin, kailangan namin ng tubig at pagkain…")
            mode_label = st.segmented_control(
                "Language model", list(MODES), default="Auto-detect",
                help="Auto-detect looks for Filipino words and picks the matching model.")
            save = st.checkbox("Save to the dashboard", value=True)
            b1, b2, _ = st.columns([2, 1, 2])
            submitted = b1.form_submit_button("Classify report", type="primary", icon="🔍",
                                              use_container_width=True)
            b2.form_submit_button("Clear", on_click=clear_form, use_container_width=True)

    result_col = st.container()   # the result shows right under the form, above the examples

    st.pills("No ideas? Try an example", list(EXAMPLES), key="example", on_change=use_example,
             help="Fills the report box with a sample. You can edit it before classifying.")

    if submitted:
        mode = MODES.get(mode_label or "Auto-detect", "auto")
        if not text.strip():
            st.session_state.pop("last", None)
            input_col.warning("Please type a report first.")
        else:
            try:
                with result_col, st.spinner("Classifying…"):
                    result = service.classify(text, mode=mode)
            except ModelUnavailableError as error:
                st.session_state.pop("last", None)
                result_col.error(str(error))
            else:
                report_id, duplicate = None, False
                if save:
                    previous = st.session_state.get("saved_texts", {})
                    key = (text.strip(), result.language_used)
                    if key in previous:
                        report_id, duplicate = previous[key], True
                    else:
                        report_id = save_report(result)
                        previous[key] = report_id
                        st.session_state.saved_texts = previous
                st.session_state.last = {"result": result, "report_id": report_id, "duplicate": duplicate}

    with result_col:
        last = st.session_state.get("last")
        if last:
            result = last["result"]
            prediction = result.prediction
            level, colour = confidence_level(prediction.confidence)
            accent = CATEGORY_COLOURS.get(prediction.category, PRIMARY)
            st.markdown(f"<style>.st-key-result_card {{--bl-accent: {accent};}}</style>",
                        unsafe_allow_html=True)

            for message in result.warnings:
                st.warning(message)

            with st.container(border=True, key="result_card"):
                st.markdown('<div class="bl-kicker">Predicted category</div>', unsafe_allow_html=True)
                icon = CATEGORY_ICONS.get(prediction.category, "📌")
                st.markdown(f'<div class="bl-category">{icon} {prediction.readable}</div>',
                            unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{prediction.confidence:.0%}")
                c1.markdown(f":{colour}-badge[{level} confidence]")
                c2.metric("Model used", LANGUAGE_NAMES.get(result.language_used, "English"))

                if result.low_confidence:
                    st.warning("Low confidence: the model is unsure about this report. "
                               "A person should review it.", icon="🧐")

                st.markdown("**Top predictions**")
                for label, probability in prediction.top:
                    st.progress(min(max(probability, 0.0), 1.0),
                                text=f"{config.readable_label(label)} · {probability:.0%}")

                d = result.detection
                if result.detection_method == "auto":
                    chips = "".join(f'<span class="bl-chip">{w}</span>' for w in d.markers[:10])
                    st.caption(f"Detected language: **{d.language.title()}** · "
                               f"{d.filipino_ratio:.0%} Filipino words")
                    if chips:
                        st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption(f"Language chosen manually: **{result.language_used.title()}** "
                               f"(auto-detect would have picked {d.language.title()})")

            if last["duplicate"]:
                st.info(f"This report was already saved as #{last['report_id']}, so it wasn't saved again.")
            elif last["report_id"] is not None:
                st.success(f"Saved as report #{last['report_id']}. See it on the Dashboard tab.", icon="✅")

with dashboard_tab:
    reports = fetch_reports()

    if reports.empty:
        with st.container(border=True):
            st.markdown("#### 📭 No reports yet")
            st.caption("Reports you classify with *Save to the dashboard* ticked will appear here "
                       "as charts and a table.")
    else:
        reports["model_name"] = reports["language_mode"].map(LANGUAGE_NAMES).fillna("Unknown")
        reports["needs_review"] = reports["confidence"].fillna(0) < config.LOW_CONFIDENCE_THRESHOLD

        with st.expander("🔎 Filters", expanded=False):
            f1, f2 = st.columns([1, 2])
            chosen_languages = f1.multiselect("Model", sorted(reports["model_name"].unique()),
                                              placeholder="All models")
            chosen_categories = f2.multiselect("Category", sorted(reports["category_name"].dropna().unique()),
                                               placeholder="All categories")
            f3, f4 = st.columns([2, 1], vertical_alignment="bottom")
            min_conf = f3.slider("Min. confidence", 0, 100, 0, 5, format="%d%%")
            review_only = f4.toggle("Needs review only")

        view = reports
        if chosen_languages:
            view = view[view["model_name"].isin(chosen_languages)]
        if chosen_categories:
            view = view[view["category_name"].isin(chosen_categories)]
        if min_conf:
            view = view[view["confidence"].fillna(0) * 100 >= min_conf]
        if review_only:
            view = view[view["needs_review"]]

        filtered = len(view) != len(reports)
        if filtered:
            st.caption(f"Showing **{len(view)}** of {len(reports)} reports (filters on).")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reports", len(view), border=True)
        avg = view["confidence"].mean()
        m2.metric("Avg. confidence", f"{avg:.0%}" if pd.notna(avg) else "—", border=True)
        m3.metric("Need review", int(view["needs_review"].sum()), border=True)
        taglish_share = (view["language_mode"] == "taglish").mean() if len(view) else 0
        m4.metric("Taglish reports", f"{taglish_share:.0%}", border=True)

        if view.empty:
            st.info("No reports match these filters.")
        else:
            ch1, ch2 = st.columns([3, 2])
            with ch1, st.container(border=True):
                section("By category")
                counts = view["category_name"].value_counts()
                colours = {config.readable_label(k): v for k, v in CATEGORY_COLOURS.items()}
                st.altair_chart(bar_chart(counts, colours={k: colours.get(k, PRIMARY) for k in counts.index}),
                                use_container_width=True)
                st.caption(f"Most common: **{counts.index[0]}**")
            with ch2, st.container(border=True):
                section("By model")
                st.altair_chart(bar_chart(view["model_name"].value_counts(), 120), use_container_width=True)
                low = int(view["needs_review"].sum())
                st.caption(f"**{low}** report(s) below {config.LOW_CONFIDENCE_THRESHOLD:.0%} confidence "
                           "need a person to check them." if low else "All reports are above the review threshold.")

            with st.container(border=True):
                section("All reports")
                table = view.assign(
                    submitted=pd.to_datetime(view["submitted_at"], errors="coerce"),
                    review=view["needs_review"].map({True: "⚠️ Review", False: ""}),
                    category=view["predicted_category"].map(CATEGORY_ICONS).fillna("📌") + " " + view["category_name"],
                )[["id", "submitted", "category", "confidence", "review", "report_text", "model_name"]]
                st.dataframe(
                    table, hide_index=True, use_container_width=True,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", width="small"),
                        "submitted": st.column_config.DatetimeColumn("Submitted", format="MMM D, h:mm a"),
                        "category": st.column_config.TextColumn("Category"),
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence", format="percent", min_value=0.0, max_value=1.0),
                        "review": st.column_config.TextColumn("Review", width="small"),
                        "report_text": st.column_config.TextColumn("Report", width="large"),
                        "model_name": st.column_config.TextColumn("Model", width="small"),
                    })

                a1, a2 = st.columns([1, 2])
                export = view.drop(columns=["model_name", "needs_review"])
                export["top_predictions"] = export["top_predictions"].apply(
                    lambda v: "; ".join(f"{config.readable_label(l)} {p:.2f}" for l, p in json.loads(v)) if v else "")
                a1.download_button("Download CSV", export.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="bantayligtas_reports.csv", mime="text/csv", icon="⬇️",
                                   use_container_width=True)

                with a2.expander("🗑️ Delete a report"):
                    labels = {int(r.id): f"#{r.id} · {r.category_name} · {str(r.report_text)[:60]}"
                              for r in view.itertuples()}
                    target = st.selectbox("Report", list(labels), format_func=labels.get)
                    confirm = st.checkbox("Yes, delete this report permanently")
                    if st.button("Delete report", disabled=not confirm):
                        delete_report(int(target))
                        st.toast(f"Deleted report #{target}.")
                        st.rerun()

with about_tab:
    a1, a2 = st.columns([3, 2], gap="large")
    with a1:
        st.markdown(f"""
#### What it does
BantayLigtas sorts disaster-related reports into the **10 HumAID humanitarian categories**
so responders can see at a glance what people need.

#### How a report is classified
1. The text is checked for Filipino words to decide whether it is English or Taglish
   (you can override this with the *Language model* selector).
2. **English reports** go to a TF-IDF (word + character n-grams) Logistic Regression model
   trained on HumAID Event Set 1 (30,381 tweets).
3. **Taglish / Filipino reports** go to a model built on frozen
   `{config.ENCODER_NAME}` sentence embeddings, adapted with 100 hand-written Taglish examples.
4. The app shows the category, the model's confidence and the top 3 alternatives.
   Below {config.LOW_CONFIDENCE_THRESHOLD:.0%} confidence a report is flagged for human review.

#### Evaluation (from the project notebooks)

| Model | Data | Accuracy |
|---|---|---|
| English TF-IDF + LogReg | HumAID dev | 0.749 |
| English TF-IDF + LogReg | HumAID test | 0.744 |
| Taglish, before adaptation | 50 held-out Taglish reports | 0.68 |
| Taglish, after adaptation | 50 held-out Taglish reports | 0.78 |

#### Limitations
- The Taglish evaluation set is small (50 reports) and was written by the project team.
- Language detection is a word-list heuristic; very short reports may be misdetected.
- Confidence is the model's own probability, not a guarantee of correctness.
- This is a research prototype: it does not contact emergency services.
""")
    with a2:
        with st.container(border=True):
            st.markdown("#### Categories")
            st.markdown("\n".join(f"- {CATEGORY_ICONS.get(k, '📌')} {v}" for k, v in config.LABELS.items()))
        with st.container(border=True):
            st.markdown("""
#### Project team
**Proponents**  
JP. Artiola · A. Ledesma · NM. Petallana · VA. Sison · AJ. Dawal

**Adviser**  
Rosmina Joy M. Cabauatan

**Institution**  
Technological Institute of the Philippines, Quezon City
""")

st.markdown(f"""
<div class="bl-footer">
  BantayLigtas v{__version__} · Research prototype. It does not contact or dispatch emergency services.<br>
  <b>In an emergency call 911.</b>
</div>
""", unsafe_allow_html=True)
