@echo off
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    py -3 -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing / checking packages (first run takes a few minutes)...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
python scripts\check_setup.py
streamlit run app.py
pause
