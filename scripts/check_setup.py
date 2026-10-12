"""Check that everything needed to run BantayLigtas is in place.

Usage:  python scripts/check_setup.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bantayligtas import config  # noqa: E402

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
problems = 0


def version(module: str):
    try:
        return importlib.import_module(module).__version__
    except ImportError:
        return None


print(f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print(FAIL, "Python 3.10 or newer is required (3.11/3.12 recommended)."); problems += 1

for module, required in (("streamlit", True), ("sklearn", True), ("numpy", True), ("pandas", True),
                         ("joblib", True), ("torch", False), ("transformers", False)):
    v = version(module)
    if v is None:
        tag = FAIL if required else WARN
        note = "" if required else " (only needed for the Taglish model)"
        print(tag, f"{module} not installed{note}")
        problems += required
    else:
        print(OK, f"{module} {v}")

sk = version("sklearn")
if sk and sk != "1.6.1":
    print(WARN, f"scikit-learn {sk}: the saved models were made with 1.6.1; "
                "loading may warn or fail. Run: pip install scikit-learn==1.6.1")
np_v = version("numpy")
if np_v and int(np_v.split(".")[0]) < 2:
    print(FAIL, f"numpy {np_v}: the saved models need numpy 2.x"); problems += 1

print(f"\nModels folder: {config.MODEL_DIR}")
for name in (config.ENGLISH_VECTORIZER_FILE, config.ENGLISH_CLASSIFIER_FILE, config.TAGLISH_CLASSIFIER_FILE):
    path = config.MODEL_DIR / name
    required = name != config.TAGLISH_CLASSIFIER_FILE
    if path.exists():
        print(OK, f"{name} ({path.stat().st_size / 1024**2:.1f} MB)")
    else:
        print(FAIL if required else WARN, f"{name} missing")
        problems += required

from bantayligtas.classifiers import ClassificationService  # noqa: E402

service = ClassificationService()
if service.english.available():
    try:
        r = service.classify("Please send food and water to the evacuation center.", mode="english")
        print(OK, f"English model works -> {r.prediction.readable} ({r.prediction.confidence:.0%})")
    except Exception as error:  # noqa: BLE001
        print(FAIL, f"English model failed to load: {error}"); problems += 1

if service.taglish.available() and service.taglish.libraries_installed():
    print("Loading Taglish model (first time downloads ~440 MB from Hugging Face)…")
    try:
        r = service.classify("May pamilya na na-trap sa second floor. Need rescue po.", mode="taglish")
        print(OK, f"Taglish model works -> {r.prediction.readable} ({r.prediction.confidence:.0%})")
    except Exception as error:  # noqa: BLE001
        print(WARN, f"Taglish model failed: {error} (the app will fall back to English)")

print("\nAll good. Start the app with:  streamlit run app.py" if not problems
      else f"\n{problems} problem(s) to fix before running the app.")
sys.exit(1 if problems else 0)
