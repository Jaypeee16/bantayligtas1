"""Central configuration for BantayLigtas.

Every path can be overridden with an environment variable so the app can be
moved around without editing code.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("BANTAYLIGTAS_MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.environ.get("BANTAYLIGTAS_DB", BASE_DIR / "bantayligtas.db"))

# Model files (same names as in the Google Drive presentation_app/models folder)
ENGLISH_VECTORIZER_FILE = "bantayligtas_tfidf.pkl"
ENGLISH_CLASSIFIER_FILE = "bantayligtas_classifier.pkl"
TAGLISH_CLASSIFIER_FILE = "dost_asti_taglish_adapted_v1.pkl"

# Frozen Filipino encoder used for the Taglish model (notebook 02)
ENCODER_NAME = os.environ.get("BANTAYLIGTAS_ENCODER", "dost-asti/RoBERTa-tl-cased")
MAX_LENGTH = 128

# Predictions below this confidence get a "please review" warning
LOW_CONFIDENCE_THRESHOLD = 0.50

# The 10 HumAID categories, in the readable form used by the dashboard
LABELS: dict[str, str] = {
    "caution_and_advice": "Caution and Advice",
    "displaced_people_and_evacuations": "Displaced People and Evacuations",
    "infrastructure_and_utility_damage": "Infrastructure and Utility Damage",
    "injured_or_dead_people": "Injured or Dead People",
    "missing_or_found_people": "Missing or Found People",
    "not_humanitarian": "Not Humanitarian",
    "other_relevant_information": "Other Relevant Information",
    "requests_or_urgent_needs": "Requests or Urgent Needs",
    "rescue_volunteering_or_donation_effort": "Rescue, Volunteering or Donations",
    "sympathy_and_support": "Sympathy and Support",
}


def readable_label(label: str) -> str:
    """Turn a raw class label into its display name (falls back gracefully)."""
    return LABELS.get(label, label.replace("_", " ").title())
