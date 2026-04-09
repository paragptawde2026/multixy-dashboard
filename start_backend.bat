@echo off
echo ============================================
echo  Starting Multi X-Y Backend (FastAPI)
echo ============================================

cd /d "%~dp0backend"

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python packages...
pip install -r requirements.txt

REM Start the FastAPI server
echo.
echo Backend running at: http://localhost:8000
echo API docs at:        http://localhost:8000/docs
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
