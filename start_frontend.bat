@echo off
echo ============================================
echo  Starting Multi X-Y Frontend (React + Vite)
echo ============================================

cd /d "%~dp0frontend"

REM Install node packages if not already done
if not exist node_modules (
    echo Installing npm packages...
    npm install
)

echo.
echo Frontend running at: http://localhost:5173
echo.
npm run dev

pause
