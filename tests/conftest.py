"""Shared fixtures: tiny stand-in models with the same structure as the real ones."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantayligtas import config  # noqa: E402
from bantayligtas.classifiers import TaglishClassifier  # noqa: E402
from bantayligtas.preprocessing import preprocess_english  # noqa: E402

EMBED_DIM = 16


def fake_embed(texts):
    """Deterministic pseudo-embeddings standing in for the RoBERTa encoder."""
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(rng.normal(size=EMBED_DIM))
    return np.array(out)


class FakeTaglish(TaglishClassifier):
    """Uses the real saved classifier file but a fake encoder (no torch needed)."""

    @staticmethod
    def libraries_installed() -> bool:
        return True

    def load(self):
        if self._classifier is None:
            if not self.file.exists():
                from bantayligtas.classifiers import ModelUnavailableError
                raise ModelUnavailableError("missing")
            self._classifier = joblib.load(self.file)

    def embed(self, texts):
        return fake_embed(texts)


@pytest.fixture(scope="session")
def training_data():
    frames = [pd.read_csv(config.DATA_DIR / f) for f in
              ("taglish_adaptation_v1.csv", "taglish_heldout_v1.csv", "taglish_diagnostic_v1.csv")]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, training_data):
    d = tmp_path_factory.mktemp("models")
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
    ])
    X = vec.fit_transform(training_data["tweet_text"].map(preprocess_english))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, training_data["class_label"])
    joblib.dump(vec, d / config.ENGLISH_VECTORIZER_FILE)
    joblib.dump(clf, d / config.ENGLISH_CLASSIFIER_FILE)

    tag = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    tag.fit(fake_embed(training_data["tweet_text"].tolist()), training_data["class_label"])
    joblib.dump(tag, d / config.TAGLISH_CLASSIFIER_FILE)
    return d
