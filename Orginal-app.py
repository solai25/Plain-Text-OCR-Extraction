import os

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
# 2. HARDWARE & BACKEND STATUS CHECK
# ==========================================
HAS_GPU = torch.cuda.is_available()
DEVICE_STATUS = "GPU (CUDA)" if HAS_GPU else "CPU / Apple Silicon"
BACKEND_STATUS = "vllm" if HAS_GPU else "llamacpp"

# ==========================================
# 3. LAZY LOADING MODEL MANAGERS
# ==========================================
_manager = None
_predictors = {}

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

def get_predictor(task):
    global _predictors
    if task not in _predictors:
        try:
            if task == "ocr":
                from surya.recognition import RecognitionPredictor
                _predictors[task] = RecognitionPredictor(get_manager())
            elif task == "detect":
                from surya.detection import DetectionPredictor
                _predictors[task] = DetectionPredictor() 
            elif task == "layout":
                from surya.layout import LayoutPredictor
                _predictors[task] = LayoutPredictor(get_manager())
            elif task == "table":
                from surya.table_rec import TableRecPredictor
                _predictors[task] = TableRecPredictor(get_manager())
        except Exception as e:
            raise gr.Error(f"Failed to load {task} model: {str(e)}")
    return _predictors[task]

def clean_surya_text(result_dict):
    """Extracts text from Surya's JSON, respects reading order, and strips HTML."""
    plain_text_pages = []
    
    for page in result_dict:
        page_text = []
        
        # Check if the result uses the 'blocks' format
        if 'blocks' in page:
            # Sort blocks by reading order to keep the document flow logical
            sorted_blocks = sorted(page['blocks'], key=lambda b: b.get('reading_order', 0))
            
            for block in sorted_blocks:
                if 'html' in block:
                    text = block['html']
                    # 1. Convert <br/> tags to actual Python newlines
                    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
                    # 2. Strip all remaining HTML tags (like <p>)
                    text = re.sub(r'<[^>]+>', '', text)
                    page_text.append(text.strip())
                elif 'text' in block:
                    page_text.append(block['text'].strip())
                    
        # Fallback for standard line-by-line text format
        elif 'text_lines' in page:
            for line in page['text_lines']:
                if 'text' in line:
                    page_text.append(line['text'].strip())
                    
        # Join blocks with a double newline
        plain_text_pages.append("\n\n".join(page_text))
        
    return "\n\n--- PAGE BREAK ---\n\n".join(plain_text_pages)

# ==========================================
# 4. SOPHISTICATED ERROR HANDLING & OUTPUT
# ==========================================
def save_output(task_name, result_dict):
    timestamp = int(time.time())
    filename = os.path.join(OUTPUT_DIR, f"{task_name}_result_{timestamp}.json")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        return filename
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
        predictor = get_predictor("ocr")
        img = process_image(image)
        
        predictions = predictor([img])
        result_dict = [pred.model_dump() for pred in predictions]
        
        # NEW: Extract and clean the text
        plain_text = clean_surya_text(result_dict)
        
        saved_path = save_output("ocr", result_dict)
        
        # NEW: Return the plain_text alongside the other outputs
        return plain_text, result_dict, f"✅ Success! Output stored in: {saved_path}"
    except Exception as e:
        raise gr.Error(f"OCR Processing failed: {str(e)}")

def process_detect(image):
    try:
        gr.Info("Starting Text Line Detection...")
        predictor = get_predictor("detect")
        img = process_image(image)
        
        predictions = predictor([img])
        result_dict = [pred.model_dump() for pred in predictions]
        
        saved_path = save_output("detect", result_dict)
        return result_dict, f"✅ Success! Output stored in: {saved_path}"
    except Exception as e:
        raise gr.Error(f"Detection failed: {str(e)}")

def process_layout(image):
    try:
        gr.Info("Analyzing Document Layout & Reading Order...")
        predictor = get_predictor("layout")
        img = process_image(image)
        
        predictions = predictor([img])
        result_dict = [pred.model_dump() for pred in predictions]
        
        saved_path = save_output("layout", result_dict)
        return result_dict, f"✅ Success! Output stored in: {saved_path}"
    except Exception as e:
        raise gr.Error(f"Layout Processing failed: {str(e)}")

def process_table(image, extract_html):
    try:
        gr.Info("Recognizing Table Structures...")
        predictor = get_predictor("table")
        img = process_image(image)
        
        if extract_html:
             predictions = predictor.predict_full([img])
        else:
             predictions = predictor([img])
             
        result_dict = [pred.model_dump() for pred in predictions]
        saved_path = save_output("table", result_dict)
        return result_dict, f"✅ Success! Output stored in: {saved_path}"
    except Exception as e:
        raise gr.Error(f"Table Recognition failed: {str(e)}")

# ==========================================
# 5. GRADIO APP INTERFACE
# ==========================================
custom_css = """
.container { max-width: 1300px; margin: auto; }
.header { text-align: center; margin-bottom: 25px; }
.header h1 { margin-bottom: 5px; color: #333; }
.footer { text-align: center; margin-top: 30px; font-size: 0.95em; color: gray; border-top: 1px solid #ddd; padding-top: 15px;}
.device-badge { background: #e0f7fa; color: #006064; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9em; display: inline-block; margin-top: 10px; }
"""

# FIX: Removed `css=custom_css` from the Blocks constructor
with gr.Blocks(title="Surya OCR Workspace") as app:
    with gr.Column(elem_classes="container"):
        # HEADER
        gr.Markdown(
            f"""
            <div class="header">
                <h1>📄 Advanced Document Intelligence</h1>
                <p>State-of-the-Art OCR, Layout Analysis, Reading Order, and Table Recognition</p>
                <div class="device-badge">Hardware: {DEVICE_STATUS} | Backend: {BACKEND_STATUS}</div>
            </div>
            """
        )
        
        # TAB 1: OCR
        with gr.Tab("📝 Text Recognition (OCR)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("Upload a high-res image of your document to extract all text.")
                    ocr_input = gr.Image(label="Input Document", type="numpy")
                    ocr_langs = gr.Textbox(label="Target Languages", info="Optional: Comma-separated language codes.", placeholder="en")
                    ocr_btn = gr.Button("Generate OCR", variant="primary")
                with gr.Column(scale=1):
                    ocr_status = gr.Textbox(label="Process Status", interactive=False)
                    
                    # NEW: A large textbox with a copy button for the clean text
                    ocr_text_output = gr.Textbox(label="Extracted Plain Text", lines=15, interactive=False)
                    
                    # NEW: Tucked the raw JSON inside a collapsed accordion
                    with gr.Accordion("View Raw JSON Data", open=False):
                        ocr_json_output = gr.JSON(label="Structured Output")
            
            gr.Examples(
                examples=[["dummy_example.jpg", "en"]],
                inputs=[ocr_input, ocr_langs],
                label="Examples",
                run_on_click=False 
            )
            
            # NEW: Make sure we output to all three UI elements
            ocr_btn.click(
                process_ocr, 
                inputs=[ocr_input, ocr_langs], 
                outputs=[ocr_text_output, ocr_json_output, ocr_status]
            )

        # TAB 2: Text Line Detection
        with gr.Tab("🔍 Line Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    # FIX: Removed `info="..."` argument from gr.Image
                    gr.Markdown("Upload an image to detect bounding boxes around text lines.")
                    detect_input = gr.Image(label="Input Document", type="numpy")
                    detect_btn = gr.Button("Detect Text Lines", variant="primary")
                with gr.Column(scale=1):
                    detect_status = gr.Textbox(label="Process Status", interactive=False)
                    detect_output = gr.JSON(label="Detection Bounding Boxes (JSON)")
                    
            gr.Examples(
                examples=[["dummy_example.jpg"]],
                inputs=[detect_input],
                label="Examples (Click an image to load it into the form, then press Generate)",
                run_on_click=False
            )
            detect_btn.click(process_detect, inputs=[detect_input], outputs=[detect_output, detect_status])

        # TAB 3: Layout Analysis
        with gr.Tab("📑 Layout & Reading Order"):
            with gr.Row():
                with gr.Column(scale=1):
                    # FIX: Removed `info="..."` argument from gr.Image
                    gr.Markdown("Upload to parse layout semantics (Headers, Tables, Graphics) and Reading Order.")
                    layout_input = gr.Image(label="Input Document", type="numpy")
                    layout_btn = gr.Button("Analyze Layout", variant="primary")
                with gr.Column(scale=1):
                    layout_status = gr.Textbox(label="Process Status", interactive=False)
                    layout_output = gr.JSON(label="Layout Analysis (JSON)")
                    
            gr.Examples(
                examples=[["dummy_example.jpg"]],
                inputs=[layout_input],
                label="Examples (Click an image to load it into the form, then press Generate)",
                run_on_click=False
            )
            layout_btn.click(process_layout, inputs=[layout_input], outputs=[layout_output, layout_status])

        # TAB 4: Table Recognition
        with gr.Tab("📊 Table Recognition"):
            with gr.Row():
                with gr.Column(scale=1):
                    # FIX: Removed `info="..."` argument from gr.Image
                    gr.Markdown("Upload an image containing a table.")
                    table_input = gr.Image(label="Input Document", type="numpy")
                    table_html = gr.Checkbox(label="Extract Full HTML", info="Check this for complete HTML output (better for merged cells). Uncheck for raw row/col data.", value=True)
                    table_btn = gr.Button("Recognize Table", variant="primary")
                with gr.Column(scale=1):
                    table_status = gr.Textbox(label="Process Status", interactive=False)
                    table_output = gr.JSON(label="Table Data (JSON)")
                    
            gr.Examples(
                examples=[["dummy_example.jpg", True]],
                inputs=[table_input, table_html],
                label="Examples (Click an image to load it into the form, then press Generate)",
                run_on_click=False
            )
            table_btn.click(process_table, inputs=[table_input, table_html], outputs=[table_output, table_status])

        # FOOTER (Credit)
        gr.Markdown(
            '''
            <div class="footer">
                <p>Created by solai | Powered by <a href="https://github.com/datalab-to/surya" target="_blank">Surya AI</a></p>
            </div>
            '''
        )

if not os.path.exists("dummy_example.jpg"):
    Image.new('RGB', (600, 400), color=(240, 240, 240)).save("dummy_example.jpg")

if __name__ == "__main__":
    # FIX: Passed `css=custom_css` into the launch method here
    app.launch(inbrowser=True, css=custom_css)
