"""Text preprocessing.

`preprocess_english` is the exact function the English TF-IDF model was
trained with in the original HumAID notebook. Notebook 03 used a slightly
different variant ("url" vs " url ", https? only); using the training version
keeps predictions consistent with the reported 0.744 test accuracy.
"""
from __future__ import annotations

import re

_URL = re.compile(r"http\S+|www\S+")
_MENTION = re.compile(r"@\w+")
_SPACES = re.compile(r"\s+")


def preprocess_english(text: str) -> str:
    t = str(text).lower()
    t = _URL.sub(" url ", t)
    t = _MENTION.sub(" user ", t)
    t = _SPACES.sub(" ", t).strip()
    return t


def clean_report(text: str) -> str:
    """Light cleanup applied to every report before storage/classification."""
    return _SPACES.sub(" ", str(text)).strip()
