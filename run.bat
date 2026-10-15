@echo off
cd /d "%~dp0"
if not exist .venv (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv || (echo Python 3.10 or newer is required. & pause & exit /b 1)
)
call .venv\Scripts\activate
pip install -q -r requirements.txt || (pause & exit /b 1)
if not exist .env copy .env.example .env >nul
python -m streamlit run app.py
pause
