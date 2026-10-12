"""Smoke test the Streamlit app end-to-end with stand-in models."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def app(model_dir, tmp_path, monkeypatch):
    import shutil
    from bantayligtas import config
    d = tmp_path / "models"
    d.mkdir()
    for f in (config.ENGLISH_VECTORIZER_FILE, config.ENGLISH_CLASSIFIER_FILE):
        shutil.copy(model_dir / f, d / f)          # English only: Taglish should fall back
    monkeypatch.setattr(config, "MODEL_DIR", d)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    import streamlit as st
    st.cache_resource.clear()
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = st_testing.AppTest.from_file(app_path, default_timeout=30)
    at.run()
    return at


def test_app_loads(app):
    assert not app.exception
    assert len(app.tabs) == 3


def test_submit_and_dashboard(app):
    app.text_area[0].input("Wala na kaming malinis na tubig dito. Tulong po!")
    app.button[0].click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Confidence", "Model used"} <= set(labels)
    assert any("Saved as report" in s.value for s in app.success)
    assert any("Taglish model unavailable" in w.value for w in app.warning)   # graceful fallback
    reports = [m for m in app.metric if m.label == "Reports"]
    assert reports and reports[0].value == "1"


def test_same_report_is_saved_once(app):
    for _ in range(2):
        app.text_area[0].input("The main bridge collapsed after the earthquake.")
        app.button[0].click().run()
    assert not app.exception
    assert any("already saved as #1" in i.value for i in app.info)
    reports = [m for m in app.metric if m.label == "Reports"]
    assert reports and reports[0].value == "1"
