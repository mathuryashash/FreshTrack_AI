@echo off
echo Starting FreshTrack API...
call venv\Scripts\activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
pause
