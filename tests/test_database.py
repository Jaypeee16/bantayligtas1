import json
import sqlite3

import pytest

from bantayligtas import database
from bantayligtas.classifiers import ClassificationService, EnglishClassifier
from tests.conftest import FakeTaglish


@pytest.fixture
def result(model_dir):
    svc = ClassificationService(EnglishClassifier(model_dir), FakeTaglish(model_dir))
    return svc.classify("May sugatan dito at kailangan ng medical assistance.")


def test_fresh_db_roundtrip(tmp_path, result):
    db = tmp_path / "new.db"
    database.init_db(db)
    rid = database.save_report(result, db)
    df = database.fetch_reports(db)
    assert len(df) == 1 and df.loc[0, "id"] == rid
    row = df.iloc[0]
    assert row["confidence"] == pytest.approx(result.prediction.confidence, abs=1e-5)
    assert row["detected_language"] == "taglish" and row["detection_method"] == "auto"
    assert len(json.loads(row["top_predictions"])) == 3
    assert row["category_name"]
    database.delete_report(rid, db)
    assert database.fetch_reports(db).empty


def test_migrates_notebook03_schema(tmp_path, result):
    db = tmp_path / "nb03.db"
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, report_text TEXT NOT NULL,
                     language_mode TEXT NOT NULL, predicted_category TEXT NOT NULL, submitted_at TEXT NOT NULL)""")
        c.execute("INSERT INTO reports (report_text, language_mode, predicted_category, submitted_at) "
                  "VALUES ('old', 'english', 'caution_and_advice', '2025-01-01')")
    database.init_db(db)
    database.save_report(result, db)
    df = database.fetch_reports(db)
    assert len(df) == 2
    old = df[df["report_text"] == "old"].iloc[0]
    assert old["detected_language"] == "english" and old["category_name"] == "Caution and Advice"


def test_migrates_notebook01_schema(tmp_path, result):
    db = tmp_path / "nb01.db"
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, report_text TEXT NOT NULL,
                     predicted_category TEXT NOT NULL, probability REAL NOT NULL, submitted_at TEXT NOT NULL)""")
        c.execute("INSERT INTO reports (report_text, predicted_category, probability, submitted_at) "
                  "VALUES ('old', 'sympathy_and_support', 0.81, '2025-01-01')")
    database.init_db(db)
    database.save_report(result, db)          # must fill the legacy NOT NULL probability column
    df = database.fetch_reports(db)
    assert len(df) == 2
    assert df[df["report_text"] == "old"].iloc[0]["confidence"] == pytest.approx(0.81)
