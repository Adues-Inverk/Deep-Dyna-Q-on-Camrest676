@echo off
title Deep Dyna-Q Dashboard
color 0A
cls
echo.
echo  ==========================================
echo   Deep Dyna-Q Demo Dashboard
echo   CamRest676 + Frames Hotel Booking
echo  ==========================================
echo.
echo  Dang khoi dong server...
echo.

cd /d "%~dp0"

call .venv\Scripts\activate.bat

echo  Mo trinh duyet tai: http://127.0.0.1:8000
echo  Nhan Ctrl+C de dung server.
echo.

start "" "http://127.0.0.1:8000"

python dashboard_server.py --no-open

pause
