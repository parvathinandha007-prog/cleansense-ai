@echo off
echo.
echo  ====================================================
echo   CleanSense AI -- Starting Application
echo  ====================================================
echo.
echo  Backend: http://localhost:8000
echo  App UI:  http://localhost:8000
echo  API docs: http://localhost:8000/docs
echo.
cd /d "%~dp0backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
