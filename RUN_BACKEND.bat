@echo off
cd /d "c:\Users\ASUS\Desktop\new gen\C LANGUAGE\c language\SIT hackthon\worldsim-backend"
echo Installing backend dependencies...
call venv\Scripts\pip install -q fastapi uvicorn numpy networkx pydantic python-dotenv PyYAML scipy matplotlib pandas
echo.
echo Starting FastAPI backend server...
call venv\Scripts\python main.py
pause
