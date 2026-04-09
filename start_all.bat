@echo off
echo ============================================
echo  Starting Multi X-Y Dashboard (Full Stack)
echo ============================================
echo.
echo Opening Backend and Frontend in separate windows...
echo.

start "Multi X-Y Backend"  cmd /k "cd /d "%~dp0" && start_backend.bat"
timeout /t 3 /nobreak >nul
start "Multi X-Y Frontend" cmd /k "cd /d "%~dp0" && start_frontend.bat"

echo.
echo Both servers are starting!
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:5173
echo  API Docs: http://localhost:8000/docs
echo.
pause
