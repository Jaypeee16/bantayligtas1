# Trained models

| File | Model | Required? |
|---|---|---|
| `bantayligtas_tfidf.pkl` | English TF-IDF vectorizer | Yes |
| `bantayligtas_classifier.pkl` | English Logistic Regression | Yes |
| `dost_asti_taglish_adapted_v1.pkl` | Taglish classifier (on DOST-ASTI RoBERTa embeddings) | Optional, falls back to English |

Saved with scikit-learn 1.6.1 and numpy 2.x. Rebuild them with `scripts/train_english.py` and
`scripts/train_taglish.py`.
