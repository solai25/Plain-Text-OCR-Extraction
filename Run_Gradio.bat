@echo off
title "OCR-App"

:: Check if the virtual environment folder already exists
if not exist .venv (
    echo Creating virtual environment...
<<<<<<< HEAD
    py -3.12 -m venv .venv
=======
    python -m venv .venv
>>>>>>> 89ed228b5e4795a57dc10a9fdbe3e533e3365047
    call .venv\Scripts\activate
    pip install -r requirements.txt
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Virtual environment already exists, skipping creation...
    call .venv\Scripts\activate
)
echo launching.....
<<<<<<< HEAD

=======
set SURYA_INFERENCE_BACKEND=llamacpp
>>>>>>> 89ed228b5e4795a57dc10a9fdbe3e533e3365047
app.py
pause