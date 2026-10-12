"""Model loading and prediction.

* EnglishClassifier  - TF-IDF (word 1-2 + char 3-5) + Logistic Regression
                       from the original HumAID notebook (test acc 0.744).
* TaglishClassifier  - frozen dost-asti/RoBERTa-tl-cased mean-pooled embeddings
                       + StandardScaler + Logistic Regression, adapted with 100
                       Taglish examples (held-out Taglish acc 0.68 -> 0.78).
* ClassificationService - picks the model automatically from the detected
                       language (or a manual choice) and returns confidence
                       scores. Falls back to English if the Taglish model
                       cannot be loaded (e.g. torch not installed).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import config
from .language import LanguageResult, detect_language
from .preprocessing import clean_report, preprocess_english


class ModelUnavailableError(RuntimeError):
    """Raised when a model's files or libraries are missing."""


@dataclass
class Prediction:
    category: str
    confidence: float
    top: list[tuple[str, float]]   # [(label, probability), ...] best first

    @property
    def readable(self) -> str:
        return config.readable_label(self.category)


def _rank(classes, probabilities, k: int = 3) -> Prediction:
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities)[::-1]
    top = [(str(classes[i]), float(probabilities[i])) for i in order[:k]]
    return Prediction(category=top[0][0], confidence=top[0][1], top=top)


class EnglishClassifier:
    name = "english_tfidf_logreg"

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self._vectorizer = None
        self._classifier = None
        self._lock = threading.Lock()

    @property
    def files(self) -> list[Path]:
        return [self.model_dir / config.ENGLISH_VECTORIZER_FILE,
                self.model_dir / config.ENGLISH_CLASSIFIER_FILE]

    def available(self) -> bool:
        return all(p.exists() for p in self.files)

    def load(self) -> None:
        with self._lock:
            if self._classifier is not None:
                return
            missing = [p.name for p in self.files if not p.exists()]
            if missing:
                raise ModelUnavailableError(
                    f"English model file(s) missing from {self.model_dir}: {', '.join(missing)}")
            self._vectorizer = joblib.load(self.files[0])
            self._classifier = joblib.load(self.files[1])

    def predict(self, text: str) -> Prediction:
        self.load()
        features = self._vectorizer.transform([preprocess_english(text)])
        probabilities = self._classifier.predict_proba(features)[0]
        return _rank(self._classifier.classes_, probabilities)


class TaglishClassifier:
    name = "taglish_dost_asti_roberta_adapted"

    def __init__(self, model_dir: Path | None = None, encoder_name: str | None = None):
        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self.encoder_name = encoder_name or config.ENCODER_NAME
        self._tokenizer = None
        self._encoder = None
        self._classifier = None
        self._lock = threading.Lock()

    @property
    def file(self) -> Path:
        return self.model_dir / config.TAGLISH_CLASSIFIER_FILE

    def available(self) -> bool:
        return self.file.exists()

    @staticmethod
    def libraries_installed() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        with self._lock:
            if self._classifier is not None:
                return
            if not self.file.exists():
                raise ModelUnavailableError(
                    f"Taglish model file missing from {self.model_dir}: {self.file.name}")
            if not self.libraries_installed():
                raise ModelUnavailableError(
                    "Taglish model needs 'torch' and 'transformers' (pip install -r requirements.txt)")
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            self._encoder = AutoModel.from_pretrained(self.encoder_name, add_pooling_layer=False)
            self._encoder.eval()
            self._classifier = joblib.load(self.file)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled sentence embeddings, exactly as in notebook 02."""
        import torch

        encoded = self._tokenizer(texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=config.MAX_LENGTH)
        with torch.no_grad():
            output = self._encoder(**encoded, return_dict=True)
        tokens = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.cpu().numpy()

    def predict(self, text: str) -> Prediction:
        self.load()
        embedding = self.embed([text])
        probabilities = self._classifier.predict_proba(embedding)[0]
        return _rank(self._classifier.classes_, probabilities)


@dataclass
class ClassificationResult:
    text: str
    prediction: Prediction
    detection: LanguageResult
    detected_language: str      # what the detector said
    detection_method: str       # "auto" or "manual"
    language_used: str          # which language model actually ran
    model_used: str
    warnings: list[str] = field(default_factory=list)

    @property
    def low_confidence(self) -> bool:
        return self.prediction.confidence < config.LOW_CONFIDENCE_THRESHOLD


class ClassificationService:
    def __init__(self, english: EnglishClassifier | None = None,
                 taglish: TaglishClassifier | None = None):
        self.english = english or EnglishClassifier()
        self.taglish = taglish or TaglishClassifier()

    def status(self) -> dict:
        return {
            "english_files": self.english.available(),
            "taglish_files": self.taglish.available(),
            "taglish_libraries": self.taglish.libraries_installed(),
            "model_dir": str(self.english.model_dir),
        }

    def classify(self, text: str, mode: str = "auto") -> ClassificationResult:
        text = clean_report(text)
        if not text:
            raise ValueError("Report text is empty.")
        if mode not in {"auto", "english", "taglish"}:
            raise ValueError(f"Unknown mode: {mode}")

        detection = detect_language(text)
        wanted = detection.language if mode == "auto" else mode
        warnings: list[str] = []

        if wanted == "taglish":
            try:
                prediction = self.taglish.predict(text)
                return ClassificationResult(text, prediction, detection, detection.language,
                                            "auto" if mode == "auto" else "manual",
                                            "taglish", self.taglish.name, warnings)
            except ModelUnavailableError as error:
                warnings.append(f"Taglish model unavailable, used the English model instead. ({error})")

        prediction = self.english.predict(text)
        return ClassificationResult(text, prediction, detection, detection.language,
                                    "auto" if mode == "auto" else "manual",
                                    "english", self.english.name, warnings)
