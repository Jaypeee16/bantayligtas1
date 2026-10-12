# BantayLigtas

BantayLigtas classifies disaster reports in English, Filipino and Taglish into the **10 HumAID humanitarian categories**. It automatically picks the right language model, shows how confident each prediction is, and keeps every report in a local dashboard.

This version combines the four project notebooks into one app that runs on your own computer, without Colab or Google Drive:

| Notebook | What it did | Where it lives now |
|---|---|---|
| `Original_HumAID_notebook` | English TF-IDF model (test accuracy 0.744) | `bantayligtas/classifiers.py` → `EnglishClassifier`, `scripts/train_english.py` |
| `02_Taglish_Language_Analysis` | DOST-ASTI RoBERTa model adapted for Taglish (0.68 → 0.78) | `TaglishClassifier`, `scripts/train_taglish.py`, `data/*.csv` |
| `03_BantayLigtas_Streamlit_Development` | Streamlit app and SQLite database | `app.py`, `bantayligtas/database.py` |
| `04_BantayLigtas_Presentation_Launcher` | Launched the app in Colab | `run.bat` / `run.sh` |

**New in this version**
- **Automatic language detection.** The app looks for Filipino words and sends each report to the English or Taglish model. You can still choose the model yourself with the *Language model* selector.
- **Confidence scores.** Each result shows its confidence and the top 3 categories. Reports below 50% confidence are flagged for human review.
- **Fixes from the notebooks.**
  - The app now uses the same preprocessing the English model was trained with.
  - Probabilities are saved again; the notebook 03 app had dropped them.
  - Old `bantayligtas.db` files from earlier versions are upgraded automatically.
  - If the Taglish model can't load, the app falls back to the English model instead of crashing.

---

## 1. Requirements

- **Python 3.11 or 3.12.** 3.10 and 3.13 also work. On Windows, tick "Add Python to PATH" when installing.
- About 3 GB of free disk space for PyTorch and the Taglish encoder.
- Internet access the first time you run it, to download packages and the 440 MB DOST-ASTI encoder from Hugging Face.

## 2. Model files

The trained models are already included in **`models/`**:

```
models/bantayligtas_tfidf.pkl              English TF-IDF vectorizer
models/bantayligtas_classifier.pkl         English Logistic Regression
models/dost_asti_taglish_adapted_v1.pkl    Taglish classifier (on DOST-ASTI RoBERTa embeddings)
```

To rebuild them from the HumAID data, see section 6.

## 3. Run it

**Windows:** double-click **`run.bat`**.
**macOS / Linux:** run `./run.sh` in a terminal.

The first run creates a virtual environment (`.venv`), installs the packages, checks your setup and opens the app at **http://localhost:8501**. Later runs start in seconds.

To do it by hand instead:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py
streamlit run app.py
```

> **English only?** You can skip `torch` and `transformers` (the last two lines of `requirements.txt`). The app still runs and sends every report to the English model.

### Put it online (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (this project lives at github.com/Jaypeee16/bantayligtas).
2. Go to **share.streamlit.io**, sign in with GitHub and click **Create app → Deploy a public app from GitHub**.
3. Pick the repository, branch `main` and main file `app.py`. Under **Advanced settings** choose **Python 3.12**.
4. Click **Deploy**. The first build takes about 5–10 minutes (it installs PyTorch). The first Taglish
   report then takes 30–60 seconds while the encoder downloads; later ones are fast.

Notes: the online database is reset whenever the app restarts or wakes from sleep, and apps sleep after
a period without visitors (open the link once before a demo to wake it up). The Taglish model needs
about 1–1.5 GB of memory; if the app runs out of memory, use the English-only setup described in section 3.

## 4. Using the app

- **Submit report.** Type or pick an example, then press *Classify report*. You'll see:
  - the category
  - the confidence
  - which model ran
  - which Filipino words triggered the language detection
  - the top 3 categories
- **Dashboard.** Shows totals, average confidence, the share of Taglish reports and charts. You can:
  - filter by model, category, minimum confidence or reports needing review
  - download the reports as CSV
  - delete reports
- **About.** Explains how it works, gives the evaluation results and lists limitations.

The first Taglish report takes about 10–30 seconds while the encoder loads. After that it is fast.

Reports are stored in `bantayligtas.db` in the project folder. To reuse a database from the old app, copy it here; it is upgraded automatically.

## 5. Project layout

```
app.py                     Streamlit interface
bantayligtas/
  config.py                paths, labels, thresholds (override with env vars)
  preprocessing.py         training-time English preprocessing
  language.py              English vs Taglish detection
  classifiers.py           English + Taglish models, auto-routing, confidence
  database.py              SQLite storage + migration of old databases
scripts/
  check_setup.py           checks versions and models
  train_english.py         rebuilds the English model from HumAID
  train_taglish.py         rebuilds the adapted Taglish model
data/
  taglish_adaptation_v1.csv   100 Taglish training examples (notebook 02)
  taglish_heldout_v1.csv      50 held-out Taglish test reports
  taglish_diagnostic_v1.csv   20 diagnostic examples
models/                    trained model files (.pkl)
tests/                     pytest suite (27 tests)
```

Environment variables (optional): `BANTAYLIGTAS_MODEL_DIR`, `BANTAYLIGTAS_DB`, `BANTAYLIGTAS_ENCODER`.

## 6. Retraining the models

You need the HumAID Event Set 1 folder: the `*_train.tsv`, `*_dev.tsv` and `*_test.tsv` files.

```bash
python scripts/train_english.py --humaid-dir path/to/events_set1    # a few minutes
python scripts/train_taglish.py --humaid-dir path/to/events_set1    # 30-90 min on CPU (embeds 30k tweets)
```

Both scripts print accuracy and macro-F1 and save the models into `models/`. The Taglish script caches embeddings, so re-runs with a different `--weight` are quick.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `InconsistentVersionWarning` or errors loading `.pkl` | `pip install scikit-learn==1.6.1`. The models were saved with that version. |
| `No module named 'numpy._core'` | The models need numpy 2.x: `pip install "numpy>=2.0,<2.3"` |
| "Taglish model unavailable" warning | Install `torch` and `transformers`, or put `dost_asti_taglish_adapted_v1.pkl` in `models/`. |
| Taglish model fails on first use | It downloads the encoder from Hugging Face, so check your internet connection. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

Run the tests with `pip install -r requirements-dev.txt` and then `pytest`. The tests use small stand-in models, so they don't need the real `.pkl` files or torch.

## Limitations

- The Taglish evaluation set is small (50 reports written by the team), so treat 0.78 as indicative.
- Language detection uses a word list. Very short reports, or English names such as "Ang Lee", can be misdetected. Use the manual selector when that happens.
- Confidence is the model's own probability, not a guarantee.
- This is a research prototype. **It does not contact or dispatch emergency services.**
