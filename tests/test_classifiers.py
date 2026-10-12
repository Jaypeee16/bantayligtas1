import pytest

from bantayligtas import config
from bantayligtas.classifiers import ClassificationService, EnglishClassifier, ModelUnavailableError
from tests.conftest import FakeTaglish


@pytest.fixture
def service(model_dir):
    return ClassificationService(EnglishClassifier(model_dir), FakeTaglish(model_dir))


def test_auto_routes_english(service):
    r = service.classify("Please send drinking water to the stranded families.")
    assert r.language_used == "english" and r.detection_method == "auto"
    assert r.prediction.category in config.LABELS


def test_auto_routes_taglish(service):
    r = service.classify("May pamilya na na-trap sa second floor. Need rescue po.")
    assert r.language_used == "taglish" and r.model_used == FakeTaglish.name


def test_manual_override(service):
    r = service.classify("May pamilya na na-trap sa second floor.", mode="english")
    assert r.language_used == "english" and r.detection_method == "manual"
    assert r.detected_language == "taglish"


def test_confidence_and_top3(service):
    p = service.classify("Nasira ang tulay at hindi madaanan.").prediction
    assert 0 < p.confidence <= 1
    assert len(p.top) == 3
    assert p.top[0] == (p.category, p.confidence)
    probs = [x[1] for x in p.top]
    assert probs == sorted(probs, reverse=True)


def test_training_examples_mostly_recovered(service, training_data):
    sample = training_data.sample(40, random_state=0)
    hits = sum(service.classify(t, mode="english").prediction.category == y
               for t, y in zip(sample["tweet_text"], sample["class_label"]))
    assert hits >= 30


def test_fallback_when_taglish_missing(model_dir, tmp_path):
    import shutil
    for f in (config.ENGLISH_VECTORIZER_FILE, config.ENGLISH_CLASSIFIER_FILE):
        shutil.copy(model_dir / f, tmp_path / f)
    svc = ClassificationService(EnglishClassifier(tmp_path), FakeTaglish(tmp_path))
    r = svc.classify("Wala na kaming tubig dito, tulong po.")
    assert r.language_used == "english" and r.warnings
    assert r.detected_language == "taglish"


def test_missing_english_raises(tmp_path):
    svc = ClassificationService(EnglishClassifier(tmp_path), FakeTaglish(tmp_path))
    with pytest.raises(ModelUnavailableError):
        svc.classify("Bridge collapsed.")


def test_rejects_empty_and_bad_mode(service):
    with pytest.raises(ValueError):
        service.classify("   ")
    with pytest.raises(ValueError):
        service.classify("hello", mode="klingon")
