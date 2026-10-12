"""Retrain the English model exactly as in the original HumAID notebook.

Usage:
    python scripts/train_english.py --humaid-dir path/to/HumAID/events_set1

The folder must contain the *_train.tsv, *_dev.tsv and *_test.tsv files
(columns: tweet_id, tweet_text, class_label). Searched recursively.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion

from bantayligtas import config
from bantayligtas.preprocessing import preprocess_english


def load_split(humaid_dir: Path, split: str) -> pd.DataFrame:
    files = sorted(humaid_dir.rglob(f"*_{split}.tsv"))
    if not files:
        raise SystemExit(f"No *_{split}.tsv files found under {humaid_dir}")
    frame = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    frame = frame.dropna(subset=["tweet_text", "class_label"])
    print(f"{split:5s}: {len(frame):,} rows from {len(files)} files")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--humaid-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    args = parser.parse_args()

    train, dev, test = (load_split(args.humaid_dir, s) for s in ("train", "dev", "test"))

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000)),
    ])
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    X_train = vectorizer.fit_transform(train["tweet_text"].map(preprocess_english))
    print("Features:", X_train.shape[1])
    classifier.fit(X_train, train["class_label"])

    for name, frame in (("dev", dev), ("test", test)):
        predictions = classifier.predict(vectorizer.transform(frame["tweet_text"].map(preprocess_english)))
        print(f"\n{name}: accuracy={accuracy_score(frame['class_label'], predictions):.4f} "
              f"macro-F1={f1_score(frame['class_label'], predictions, average='macro'):.4f}")
        if name == "test":
            print(classification_report(frame["class_label"], predictions, zero_division=0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.out_dir / config.ENGLISH_VECTORIZER_FILE)
    joblib.dump(classifier, args.out_dir / config.ENGLISH_CLASSIFIER_FILE)
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()
