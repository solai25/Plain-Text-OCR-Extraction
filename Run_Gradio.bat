@echo off
title "OCR-App"

:: Check if the virtual environment folder already exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Virtual environment already exists, skipping creation...
    call .venv\Scripts\activate
)
echo launching.....
set SURYA_INFERENCE_BACKEND=llamacpp
app.py
pause