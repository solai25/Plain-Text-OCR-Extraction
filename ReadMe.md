# Plain Text OCR Extraction - Created BY Solai - Powered by Surya OCR
                
A high-performance, web-based Document OCR extraction tool built with Gradio and powered by Surya AI. This application provides state-of-the-art text extraction from documents using local GGUF models accelerated via llama.cpp. It handles layout reading order seamlessly, strips out unnecessary HTML formatting, and saves text-only transcripts with automated GPU/CUDA acceleration support.

## 🚀 Features
* GGUF & llama.cpp Integration: Optimized to run Quantize GGUF models locally using an embedded llama-cpp engine to minimize VRAM usage while maintaining accuracy.

* GPU-Accelerated Processing: Full native integration with NVIDIA CUDA wheels for blazing-fast inference speeds.

* Intelligent Text Cleaning: Automatically parsing Surya's layout blocks, honoring reading order indices, and stripping out HTML tags to yield clean plain text.

* Automatic File Management: Automatically tracks, manages, and creates local storage directories. Saves plain text outputs with custom unix timestamps.

* Lazy Loading Architecture: Postpones loading heavy deep learning models into system memory until the first execution request is triggered.

* Dual View Display: Includes a clean human-readable plain text interface along with a collapsible raw JSON inspection panel for debugging.

* Dynamic Hardware Badge: Displays current hardware constraints (GPU (CUDA) or CPU / Apple Silicon) right inside the UI banner.

## 📁 Folder Structure
````
OCR-Surya/
│
├── .venv/                  # Python virtual environment (isolated packages)
├── llama-cpp/              # Local llama.cpp binaries directory
│   ├── llama-server.exe    # Core execution server for GGUF models (CUDA-enabled)
│   └── [other dll files]   # Extracted CUDA dependency libraries
├── models/                 # Local directory cache for Hugging Face weights (~2.5GB+)
├── outputs/                # Generated plain-text OCR transcripts (*.txt)
│
├── app.py                  # Main Gradio web application script
└── requirements.txt        # High-level UI and engine dependency file
````
## 🛠️ System Requirements
### Operating System
* Windows 10 / 11 (Native Execution on local drive recommended, e.g., D: drive)

* Linux / macOS (Supported via standard fallback paths)

### Core Versions & Constraints
* Python: 3.10 or higher recommended.

* Gradio: v6.0.0+ (Required for updated routing frameworks).

* Surya OCR: v0.20.0+ (Introduces the required SuryaInferenceManager architecture).

* PyTorch: >=2.7.0 (Strict dependency forced by Surya v0.20.0+).

* NVIDIA CUDA Drivers: Compatible with CUDA 12.1 through CUDA 12.8+ environments.

## 💻 Installation & Setup Steps
** Follow these steps carefully to ensure your local Python environment and the standalone llama.cpp binaries are correctly configured for GPU usage. ** 

### 1. Initialize Virtual Environment
Open your terminal inside the project root folder and run:
````
python -m venv .venv
.venv\Scripts\activate
````
### 2. Install GPU-Accelerated PyTorch Manually
To bypass standard Windows package registry overrides and fulfill Surya's modern requirements, install the explicit CUDA-linked wheels directly:

````
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade
````
(Note: If utilizing an older system configuration, modify `cu128` to `cu124` or `cu121` matching your native graphics driver parameters).

### 3. Install App Requirements
Ensure your local requirements.txt contains the following lines:

````Plaintext
gradio>=6.0.0
pydantic
pillow
surya-ocr>=0.20.0
pdftext
requests
````

Then run the final installation check:

````DOS
pip install -r requirements.txt
````

### 4. Configure Local llama.cpp Manually (For GGUF Execution)
Since this application uses GGUF models locally on Windows, it relies on a local llama-server.exe binary compilation.
On the first run of this app automatically did this for you, but in case if the version is not match do it manually, by following the steps.

* Create a folder named llama-cpp directly in your project root directory.

* Go to the official [llama.cpp](https://github.com/ggerganov/llama.cpp/releases) Releases Page.

* Download the CUDA Windows binary zip bundle matching your CUDA installed in your System (e.g., look for files containing win-cuda-cu12.2-x64.zip or similar).

* Extract the contents of that zip file and copy all files (including `llama-server.exe` and its associated `.dll` files) directly into your newly created `llama-cpp/` project directory. **And add that llama-cpp folder location to environment variable path.**


## ⚡ Running the Application
Launch the local web server by executing:

````DOS
python app.py
````
* The script will initialize your environments, linking HF_HOME directly to your local `./models/` folder to save system drive space.

* It will detect your local llama-cpp backend configuration to drive the GGUF text layers.

* An interactive browser tab will launch automatically at http://127.0.0.1:7860.

 * Check the interface header banner to confirm that your hardware allocation reads: Hardware: GPU (CUDA).