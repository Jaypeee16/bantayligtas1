# BantayLigtas

BantayLigtas is a Streamlit application that classifies disaster reports written in English, Filipino, or Taglish. It assigns each report to one of the 10 HumAID humanitarian categories and stores the results in a local dashboard.

## Features

- Automatic English, Filipino, and Taglish language detection
- Disaster report classification into 10 humanitarian categories
- Manual language model selection when needed
- Confidence scores and top category predictions
- Human-review flag for low-confidence results
- Dashboard with report totals, charts, filters, and CSV export
- Local SQLite storage for submitted reports
- Taglish model fallback to the English model when unavailable

## How to Run

### Windows

Double-click `run.bat`, or run this command in PowerShell:

```powershell
.\run.bat
```

The script creates the virtual environment, installs the required packages, checks the models, and starts the app at:

<http://localhost:8501>

If port 8501 is already in use, start the app on another port:

```powershell
streamlit run app.py --server.port 8502
```

### macOS or Linux

```bash
./run.sh
```

The first run requires internet access to install dependencies and download the Taglish encoder. The first Taglish prediction may take longer while the model loads.

## Manual Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Reports are saved locally in `bantayligtas.db`.
