"""Retrain the adapted Taglish model as in notebook 02.

Usage:
    python scripts/train_taglish.py --humaid-dir path/to/HumAID/events_set1

Steps: embed the HumAID training set with the frozen DOST-ASTI RoBERTa encoder
(slow on CPU: ~30k tweets, expect 30-90 minutes; a GPU helps a lot), add the
100 Taglish adaptation examples with sample weight 30, fit
StandardScaler + LogisticRegression, then report accuracy on the 50 held-out
Taglish reports and the HumAID dev set.

Embeddings are cached in --cache-dir so re-runs are fast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bantayligtas import config
from bantayligtas.classifiers import TaglishClassifier
from scripts.train_english import load_split


def embed_all(model: TaglishClassifier, texts: list[str], batch_size: int = 32) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(model.embed(texts[start:start + batch_size]))
        if (start // batch_size) % 50 == 0:
            print(f"  embedded {start + len(chunks[-1]):,}/{len(texts):,}", flush=True)
    return np.vstack(chunks)


def cached_embeddings(model, name: str, texts: list[str], cache_dir: Path) -> np.ndarray:
    path = cache_dir / f"{name}_embeddings.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(texts):
            print(f"Using cached embeddings: {path}")
            return cached
    print(f"Embedding {name} ({len(texts):,} texts)…")
    X = embed_all(model, texts)
    np.save(path, X)
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--humaid-dir", type=Path, required=True)
    parser.add_argument("--adaptation-csv", type=Path, default=config.DATA_DIR / "taglish_adaptation_v1.csv")
    parser.add_argument("--heldout-csv", type=Path, default=config.DATA_DIR / "taglish_heldout_v1.csv")
    parser.add_argument("--weight", type=float, default=30.0, help="sample weight for Taglish examples")
    parser.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=config.BASE_DIR / ".embedding_cache")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not TaglishClassifier.libraries_installed():
        raise SystemExit("Install torch and transformers first: pip install -r requirements.txt")

    encoder = TaglishClassifier()
    from transformers import AutoModel, AutoTokenizer
    encoder._tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_name)
    encoder._encoder = AutoModel.from_pretrained(encoder.encoder_name, add_pooling_layer=False).eval()

    train, dev = load_split(args.humaid_dir, "train"), load_split(args.humaid_dir, "dev")
    adapt = pd.read_csv(args.adaptation_csv)
    heldout = pd.read_csv(args.heldout_csv)

    X_train = cached_embeddings(encoder, "humaid_train", train["tweet_text"].astype(str).tolist(), args.cache_dir)
    X_dev = cached_embeddings(encoder, "humaid_dev", dev["tweet_text"].astype(str).tolist(), args.cache_dir)
    X_adapt = embed_all(encoder, adapt["tweet_text"].astype(str).tolist())
    X_held = embed_all(encoder, heldout["tweet_text"].astype(str).tolist())

    X = np.vstack([X_train, X_adapt])
    y = np.concatenate([train["class_label"].to_numpy(), adapt["class_label"].to_numpy()])
    weights = np.concatenate([np.ones(len(X_train)), np.full(len(X_adapt), args.weight)])

    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42))
    print("Training…")
    model.fit(X, y, logisticregression__sample_weight=weights)

    for name, Xe, ye in (("Taglish held-out", X_held, heldout["class_label"]),
                         ("HumAID dev", X_dev, dev["class_label"])):
        p = model.predict(Xe)
        print(f"{name}: accuracy={accuracy_score(ye, p):.4f} macro-F1={f1_score(ye, p, average='macro'):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / config.TAGLISH_CLASSIFIER_FILE)
    print("Saved to", args.out_dir / config.TAGLISH_CLASSIFIER_FILE)


if __name__ == "__main__":
    main()
