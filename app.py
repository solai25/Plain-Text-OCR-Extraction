import os
<<<<<<< HEAD
import sys
import shutil
import urllib.request
import zipfile

# =====================================================================
# AUTOMATED RUNTIME LLAMA-CPP ENVIRONMENT SETUP (PURE PYTHON)
# =====================================================================
LOCAL_LLAMA_DIR = os.path.abspath("llama_cpp")

if sys.platform == "win32":
    # Check if llama-server is already globally accessible in the system PATH
    llama_present = shutil.which("llama-server") is not None
    
    if not llama_present:
        if not os.path.exists(os.path.join(LOCAL_LLAMA_DIR, "llama-server.exe")):
            print("[INFO] Local llama-server executable not detected. Beginning automated configuration...")
            os.makedirs(LOCAL_LLAMA_DIR, exist_ok=True)
            zip_target = os.path.join(LOCAL_LLAMA_DIR, "llama_bins.zip")
            
            url = "https://github.com/ggml-org/llama.cpp/releases/download/b9873/llama-b9873-bin-win-cuda-13.3-x64.zip"
            
            
            try:
                print("Downloading binary archive...")
                urllib.request.urlretrieve(url, zip_target)
                
                print("Extracting components...")
                with zipfile.ZipFile(zip_target, 'r') as zip_ref:
                    zip_ref.extractall(LOCAL_LLAMA_DIR)
                    
                if os.path.exists(zip_target):
                    os.remove(zip_target)
                print("[SUCCESS] Local llama-server installation resolved.")
            except Exception as e:
                print(f"[ERROR] Failed to automatically download llama.cpp: {e}")
            
        # Dynamically append target binary directory to active session path block
        os.environ["PATH"] = LOCAL_LLAMA_DIR + os.pathsep + os.environ["PATH"]
=======
>>>>>>> 89ed228b5e4795a57dc10a9fdbe3e533e3365047

# ==========================================
# 1. ENVIRONMENT & FOLDER SETUP
# ==========================================
MODEL_DIR = os.path.abspath("models")
OUTPUT_DIR = os.path.abspath("outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These MUST be set before importing torch, gradio, or surya
os.environ["HF_HOME"] = MODEL_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_DIR
os.environ["HF_HUB_CACHE"] = MODEL_DIR

<<<<<<< HEAD
# FORCE SURYA TO USE NATIVE LLAMA.CPP INSTEAD OF VLLM/DOCKER
os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
# FORCE THE CUDA LLAMA-SERVER BINARY TO OFFLOAD ALL LAYERS TO GPU
os.environ["LLAMA_ARG_N_GPU_LAYERS"] = "99"

=======
>>>>>>> 89ed228b5e4795a57dc10a9fdbe3e533e3365047
# ==========================================
# 2. STANDARD IMPORTS (AFTER ENV VARIABLES)
# ==========================================
import json
import time
import traceback
from PIL import Image
import torch
import gradio as gr
import re

# ==========================================
# 3. HARDWARE & BACKEND STATUS CHECK
# ==========================================
HAS_GPU = torch.cuda.is_available()
DEVICE_STATUS = "GPU (CUDA)" if HAS_GPU else "CPU / Apple Silicon"
<<<<<<< HEAD

# FIX THE VISUAL BUG: Reflect the actual backend being used
BACKEND_STATUS = os.environ.get("SURYA_INFERENCE_BACKEND", "vllm" if HAS_GPU else "llamacpp")
=======
BACKEND_STATUS = "vllm" if HAS_GPU else "llamacpp"
>>>>>>> 89ed228b5e4795a57dc10a9fdbe3e533e3365047

# ==========================================
# 4. LAZY LOADING MODEL MANAGERS
# ==========================================
_manager = None
_predictor = None

def get_manager():
    global _manager
    if _manager is None:
        try:
            from surya.inference import SuryaInferenceManager
            _manager = SuryaInferenceManager()
        except ImportError as e:
            raise gr.Error(f"Missing dependency: {str(e)}. Please ensure vllm or llama.cpp is set up.")
        except Exception as e:
            raise gr.Error(f"Failed to start Surya Inference Manager. Error: {str(e)}")
    return _manager

def get_ocr_predictor():
    global _predictor
    if _predictor is None:
        try:
            from surya.recognition import RecognitionPredictor
            _predictor = RecognitionPredictor(get_manager())
        except Exception as e:
            raise gr.Error(f"Failed to load OCR model: {str(e)}")
    return _predictor

def clean_surya_text(result_dict):
    """Extracts text from Surya's JSON, respects reading order, and strips HTML."""
    plain_text_pages = []
    
    for page in result_dict:
        page_text = []
        if 'blocks' in page:
            sorted_blocks = sorted(page['blocks'], key=lambda b: b.get('reading_order', 0))
            for block in sorted_blocks:
                if 'html' in block:
                    text = block['html']
                    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', '', text)
                    page_text.append(text.strip())
                elif 'text' in block:
                    page_text.append(block['text'].strip())
        elif 'text_lines' in page:
            for line in page['text_lines']:
                if 'text' in line:
                    page_text.append(line['text'].strip())
                    
        plain_text_pages.append("\n\n".join(page_text))
        
    return "\n\n--- PAGE BREAK ---\n\n".join(plain_text_pages)

# ==========================================
# 5. SOPHISTICATED ERROR HANDLING & OUTPUT
# ==========================================
def save_output_txt(plain_text):
    """Saves ONLY the plain text format to the output directory."""
    timestamp = int(time.time())
    txt_filename = os.path.join(OUTPUT_DIR, f"ocr_text_{timestamp}.txt")
    
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(plain_text)
        return txt_filename
    except Exception as e:
        print(f"File Save Error: {traceback.format_exc()}")
        return None

def process_image(image_input):
    if image_input is None:
        raise gr.Error("No image provided. Please upload an image first.")
    return Image.fromarray(image_input).convert("RGB")

def process_ocr(image, langs):
    try:
        gr.Info("Waking up Surya OCR... Starting analysis.")
        predictor = get_ocr_predictor()
        img = process_image(image)
        
        predictions = predictor([img])
        result_dict = [pred.model_dump() for pred in predictions]
        
        plain_text = clean_surya_text(result_dict)
        
        # Save only the TXT file
        saved_txt = save_output_txt(plain_text)
        
        if saved_txt:
            status_msg = f"✅ Success!\nText file saved to: {saved_txt}"
        else:
            status_msg = "⚠️ Processed, but failed to save the text file to disk."
            
        return plain_text, result_dict, status_msg
    except Exception as e:
        raise gr.Error(f"OCR Processing failed: {str(e)}")

# ==========================================
# 6. GRADIO APP INTERFACE
# ==========================================
custom_css = """
.container { max-width: 1300px; margin: auto; }
.header { text-align: center; margin-bottom: 25px; }
.header h1 { margin-bottom: 5px; color: #333; }
.footer { text-align: center; margin-top: 30px; font-size: 0.95em; color: gray; border-top: 1px solid #ddd; padding-top: 15px;}
.device-badge { background: #e0f7fa; color: #006064; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9em; display: inline-block; margin-top: 10px; }
"""

with gr.Blocks(title="Surya OCR Workspace") as app:
    with gr.Column(elem_classes="container"):
        # HEADER
        gr.Markdown(
            f"""
            <div class="header">
                <h1>📄 Plain Text OCR Extraction</h1>                
                <div class="device-badge">Hardware: {DEVICE_STATUS} | Backend: {BACKEND_STATUS}</div>
            </div>
            """
        )
        
        # MAIN OCR WORKSPACE
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("Upload a high-res image of your document to extract all text.")
                ocr_input = gr.Image(label="Input Document", type="numpy")
                ocr_langs = gr.Textbox(label="Target Languages", info="Optional: Comma-separated language codes.", placeholder="en")
                ocr_btn = gr.Button("Generate OCR", variant="primary")
            with gr.Column(scale=1):
                ocr_status = gr.Textbox(label="Process Status", interactive=False, lines=3)
                
                ocr_text_output = gr.Textbox(label="Extracted Plain Text", lines=15, interactive=False)
                
                with gr.Accordion("View Raw JSON Data", open=False):
                    ocr_json_output = gr.JSON(label="Structured Output")
        
        # EVENT LISTENER
        ocr_btn.click(
            process_ocr, 
            inputs=[ocr_input, ocr_langs], 
            outputs=[ocr_text_output, ocr_json_output, ocr_status]
        )

        # FOOTER (Credit)
        gr.Markdown(
            '''
            <div class="footer">
                <p>Created by solai | Powered by <a href="https://github.com/datalab-to/surya" target="_blank">Surya AI</a></p>
            </div>
            '''
        )

if __name__ == "__main__":
    app.launch(inbrowser=True, css=custom_css)
