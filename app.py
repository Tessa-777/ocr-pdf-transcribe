"""
Hardware-Optimized Educational Toolkit for ASUS Vivobook X1502ZA

HARDWARE DETECTED & OPTIMIZATION SETTINGS:
==========================================
CPU: 12th Gen Intel Core i3-1220P
  - Logical Processors: 12
  - Cores: 10
  - OCR Jobs: 11 (Total Threads - 1)

RAM: 8.00 GB Installed (486 MB Available - Low Memory Warning)
  - Whisper Model: "small" (default) - Changed from "medium" to prevent OOM crashes
  - Compute Type: int8 (CPU-optimized)

GPU: No dedicated NVIDIA GPU detected
  - Device: CPU
  - Compute Type: int8 (for faster-whisper)

OPTIMIZATIONS APPLIED:
- Model caching (@st.cache_resource) to prevent reloading on each use
- VAD filter enabled for ~30% faster transcription (skips silent parts)
- OCR parallelization: 11 jobs (matches CPU cores)
- OCR optimization disabled (optimize=0) for speed
- Image resizing to max 1024x1024 for Ollama to prevent OOM
- Auto-check for minicpm-v model availability
"""

import streamlit as st
import os
import shutil
import importlib
import time
import re
from pathlib import Path
import yt_dlp
from faster_whisper import WhisperModel
import ocrmypdf
import fitz  # pymupdf
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import subprocess
import sys
import pandas as pd

# Try to load .env file - use python-dotenv if available, otherwise manual parsing
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / ".env"

# Try python-dotenv first
env_loaded = False
try:
    from dotenv import load_dotenv
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        env_loaded = True
    else:
        load_dotenv(override=True)
        env_loaded = True
except ImportError:
    pass  # python-dotenv not installed, will use manual parser
except Exception:
    pass  # Error loading with python-dotenv, will use manual parser

# If python-dotenv didn't work or isn't installed, manually parse .env file
if not env_loaded and env_path.exists():
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Set environment variable (override=True behavior)
                    if key and value:
                        os.environ[key] = value
    except Exception:
        pass  # Silently fail if .env can't be read

# ============================================================================
# HARDWARE CONFIGURATION (Based on System Analysis)
# ============================================================================
CPU_LOGICAL_PROCESSORS = 12  # From system info: 12 logical processors
CPU_CORES = 10  # From system info: 10 cores
OCR_JOBS = CPU_LOGICAL_PROCESSORS - 1  # 11 jobs (leave 1 core free)

RAM_TOTAL_GB = 8.0  # From system info: 8.00 GB installed
RAM_AVAILABLE_MB = 486  # From system info: 486 MB available (LOW!)

# GPU Detection
HAS_NVIDIA_GPU = False  # No dedicated GPU detected in system info
torch_spec = importlib.util.find_spec("torch")
if torch_spec:
    torch = importlib.import_module("torch")
    HAS_NVIDIA_GPU = torch.cuda.is_available()

# Device and compute type based on hardware
WHISPER_DEVICE = "cuda" if HAS_NVIDIA_GPU else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if HAS_NVIDIA_GPU else "int8"

# Default model size based on RAM (8GB total but only 486MB available = use small)
DEFAULT_WHISPER_MODEL = "small"  # Changed from "medium" to prevent crashes

# Gemini configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_DELAY_SECONDS = 4.5  # Respect free-tier rate limits (15 RPM)
SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Educational Toolkit (Optimized)",
    page_icon="📚",
    layout="wide"
)

# ============================================================================
# CONSTANTS
# ============================================================================
TEMP_FOLDER = Path("temp_files")
SUPPORTED_AUDIO_FORMAT = "m4a"
MAX_IMAGE_SIZE = 1024  # Max dimension for Ollama to prevent OOM

# Get Gemini API key from environment variable (.env file only)
# Check environment variable (loaded from .env file)
GEMINI_API_KEY_BACKEND = os.environ.get("GEMINI_API_KEY", "").strip()

# Session state defaults for memo generator so state persists across reruns
SESSION_DEFAULTS = {
    "memo_images": [],
    "memo_source_name": None,
    "memo_page_input": "",
    "memo_pages_label": "",
    # OCR batch processing
    "ocr_file_queue": [],
    "ocr_processed_files": [],
    # Memo chat context
    "memo_chat_history": [],
    "memo_context": {
        "ocr_results": {},
        "extracted_passages": {},
        "generated_memo": "",
        "page_numbers": [],
        "page_mapping": {}  # Maps displayed page numbers to PDF page numbers
    },
}
for key, value in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ============================================================================
# INITIALIZATION (run once per Streamlit session)
# ============================================================================
def init_temp_folder():
    """
    Create temp_files folder once per session. Avoid deleting between reruns
    so uploaded files remain available for the memo generator.
    """
    if not TEMP_FOLDER.exists():
        TEMP_FOLDER.mkdir(exist_ok=True)

if "temp_folder_ready" not in st.session_state:
    # Clean folder once when app starts
    if TEMP_FOLDER.exists():
        shutil.rmtree(TEMP_FOLDER)
    TEMP_FOLDER.mkdir(exist_ok=True)
    st.session_state["temp_folder_ready"] = True
else:
    init_temp_folder()

# ============================================================================
# MODEL CACHING (Critical Performance Optimization)
# ============================================================================
@st.cache_resource
def load_whisper_model(model_size: str, device: str, compute_type: str):
    """
    Load Whisper model with caching to prevent reloading on each use.
    This saves minutes of waiting time.
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        subprocess.run(["tesseract", "--version"], 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def configure_gemini(api_key: str):
    """Configure Gemini client with provided API key"""
    if not api_key:
        return False, "Set GEMINI_API_KEY in .env file to enable AI memos."
    try:
        genai.configure(api_key=api_key)
        return True, "Gemini connected."
    except Exception as e:
        return False, f"Gemini error: {str(e)}"

def resize_image_for_model(image_path: Path, max_size: int = MAX_IMAGE_SIZE) -> Path:
    """
    Resize image to max 1024x1024 to prevent OOM crashes on low-memory systems.
    Returns path to resized image.
    """
    path = Path(image_path)
    with Image.open(path) as img:
        original_size = img.size
        
        # Calculate new size maintaining aspect ratio
        if max(original_size) <= max_size:
            return path  # No resize needed
        
        ratio = max_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # Resize with high-quality resampling
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save resized image
    resized_path = TEMP_FOLDER / f"resized_{path.name}"
    img_resized.save(resized_path, "PNG", optimize=False)
    
    return resized_path

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def parse_page_input(input_string: str, max_pages: int):
    """
    Parse expressions like "1-3, 5, 10-12" into a sorted list of unique page numbers.
    Returns (pages_list, errors_list).
    """
    if not input_string:
        return [], ["Enter at least one page range."]
    
    allowed_chars = re.compile(r"^[0-9,\-\s]+$")
    if not allowed_chars.match(input_string):
        return [], ["Only digits, commas, spaces, and hyphens are allowed."]
    
    pages = set()
    errors = []
    tokens = [token.strip() for token in input_string.split(",") if token.strip()]
    
    for token in tokens:
        if "-" in token:
            try:
                start_str, end_str = [part.strip() for part in token.split("-", 1)]
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                errors.append(f"Invalid range '{token}'.")
                continue
            if start > end:
                errors.append(f"Range '{token}' is reversed.")
                continue
            for page in range(start, end + 1):
                if 1 <= page <= max_pages:
                    pages.add(page)
                else:
                    errors.append(f"Page {page} is out of bounds (1-{max_pages}).")
        else:
            try:
                page = int(token)
                if 1 <= page <= max_pages:
                    pages.add(page)
                else:
                    errors.append(f"Page {page} is out of bounds (1-{max_pages}).")
            except ValueError:
                errors.append(f"Invalid page '{token}'.")
    
    return sorted(pages), errors

def compress_pages(pages):
    """Convert sorted page list into a compact string (e.g., [1,2,3,5] -> '1-3, 5')."""
    if not pages:
        return ""
    ranges = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)

def build_memo_filename(source_name: str, pages_label: str):
    """Create a descriptive memo filename."""
    base = source_name or "Memo"
    label = pages_label or "Pages"
    raw_name = f"{base} {label} Memo".strip()
    safe_name = re.sub(r'[\\\\/:*?"<>|]+', "_", raw_name)
    return f"{safe_name}.txt"

def slugify_filename(text: str, default: str = "file"):
    """Return a filesystem-safe filename fragment."""
    text = (text or "").strip()
    if not text:
        return default
    text = re.sub(r'[\\\\/:*?"<>|]+', " ", text)
    text = re.sub(r'\s+', "_", text)
    cleaned = re.sub(r"[^\w\-.]", "", text)
    return cleaned or default

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def extract_text_from_pdf_page(pdf_path: Path, page_num: int) -> str:
    """Extract text from a specific PDF page using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num - 1 < len(doc):
            page = doc[page_num - 1]
            text = page.get_text()
            doc.close()
            return text.strip()
        doc.close()
        return ""
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_from_image(image_path: Path) -> str:
    """Extract text from an image using OCR (requires OCR'd PDF or manual extraction)."""
    # For now, return placeholder - in production, you'd use pytesseract or similar
    return "[Image text extraction - OCR results would appear here]"

def process_single_ocr_file(file_info: dict) -> dict:
    """
    Process a single file with OCR.
    Returns dict with status, output_path, error message if any.
    """
    file_path = file_info["path"]
    file_name = file_info["name"]
    file_type = file_info["type"]
    
    try:
        if file_type == "pdf":
            # Process PDF with ocrmypdf
            source_slug = slugify_filename(Path(file_name).stem, "document")
            output_filename = f"ocr_{source_slug}.pdf"
            output_path = TEMP_FOLDER / output_filename
            
            ocrmypdf.ocr(
                file_path,
                output_path,
                deskew=True,
                optimize=0,
                skip_text=True,
                jobs=OCR_JOBS,
                fast_web_view=0
            )
            
            return {
                "status": "completed",
                "output_path": str(output_path),
                "output_filename": output_filename,
                "error": None
            }
        else:
            # For images, convert to PDF first, then OCR
            # Save image temporarily
            img_path = Path(file_path)
            with Image.open(img_path) as img:
                # Convert image to PDF
                pdf_path = TEMP_FOLDER / f"temp_{Path(file_name).stem}.pdf"
                img.save(pdf_path, "PDF", resolution=300.0)
            
            # OCR the PDF
            source_slug = slugify_filename(Path(file_name).stem, "document")
            output_filename = f"ocr_{source_slug}.pdf"
            output_path = TEMP_FOLDER / output_filename
            
            ocrmypdf.ocr(
                str(pdf_path),
                output_path,
                deskew=True,
                optimize=0,
                skip_text=True,
                jobs=OCR_JOBS,
                fast_web_view=0
            )
            
            # Clean up temp PDF
            if pdf_path.exists():
                pdf_path.unlink()
            
            return {
                "status": "completed",
                "output_path": str(output_path),
                "output_filename": output_filename,
                "error": None
            }
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "output_filename": None,
            "error": str(e)
        }

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("📚 Educational Toolkit")
st.sidebar.markdown("---")

tool = st.sidebar.radio(
    "Select a Tool:",
    ["The Archive (YouTube Transcriber)", 
     "The Digitizer (PDF OCR)", 
     "The Auto-Marker (Memo Generator)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

# Hardware Info Display
st.sidebar.markdown("### 💻 Hardware Info")
st.sidebar.info(f"""
**CPU:** {CPU_CORES} cores, {CPU_LOGICAL_PROCESSORS} threads  
**RAM:** {RAM_TOTAL_GB} GB (⚠️ {RAM_AVAILABLE_MB} MB available)  
**GPU:** {'NVIDIA CUDA' if HAS_NVIDIA_GPU else 'CPU Only'}  
**OCR Jobs:** {OCR_JOBS}
""")

st.sidebar.markdown("### 🔑 Gemini API")

# Only use backend API key from .env file
if GEMINI_API_KEY_BACKEND:
    st.sidebar.success("✅ API key loaded from .env file")
    st.sidebar.caption("Configured via GEMINI_API_KEY in .env")
    gemini_status, gemini_msg = configure_gemini(GEMINI_API_KEY_BACKEND)
    if gemini_status:
        st.sidebar.success(gemini_msg)
    else:
        st.sidebar.error(gemini_msg)
else:
    st.sidebar.error("❌ API key not found")
    st.sidebar.caption("Create a .env file with: GEMINI_API_KEY=your_key")
    
    # Debug info to help diagnose
    script_dir = Path(__file__).parent.absolute()
    env_path = script_dir / ".env"
    try:
        import dotenv
        dotenv_installed = True
    except ImportError:
        dotenv_installed = False
    
    with st.sidebar.expander("🔍 Debug Info"):
        st.write(f"**Script directory:** `{script_dir}`")
        st.write(f"**.env path:** `{env_path}`")
        st.write(f"**.env exists:** {env_path.exists()}")
        st.write(f"**python-dotenv installed:** {dotenv_installed}")
        if env_path.exists():
            st.warning("⚠️ .env file exists but API key not loaded. Check the file format.")
            st.code("GEMINI_API_KEY=your_key_here", language=None)
            # Check if .env file has the key
            try:
                with open(env_path, 'r') as f:
                    content = f.read()
                    if 'GEMINI_API_KEY' in content:
                        st.success("✅ GEMINI_API_KEY found in .env file")
                    else:
                        st.error("❌ GEMINI_API_KEY not found in .env file content")
            except Exception as e:
                st.error(f"Error reading .env: {e}")
    
    st.sidebar.info("""
    **Setup Instructions:**
    1. Create a `.env` file in the project root
    2. Add: `GEMINI_API_KEY=your_api_key_here`
    3. **Restart Streamlit completely** (stop and start again)
    """)
    gemini_status = False


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
if tool == "The Archive (YouTube Transcriber)":
    st.title("🎬 The Archive - YouTube Transcriber")
    st.markdown("Download and transcribe YouTube videos to text and subtitles.")
    
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        language_option = st.selectbox(
            "Language",
            ["English (SA Accent)", "Thai", "Auto-Detect"]
        )
    
    with col2:
        speed_mode = st.radio(
            "Mode",
            ["Speed", "Accuracy"],
            help="Speed: Faster processing, lower accuracy. Accuracy: Slower but more precise."
        )
    
    with col3:
        # Model size based on speed mode
        if speed_mode == "Speed":
            model_size = "small"
            beam_size = 1
        else:
            model_size = DEFAULT_WHISPER_MODEL  # "small" (was "medium" but RAM is low)
            beam_size = 5
        
        st.info(f"Model: **{model_size}**\nBeam Size: **{beam_size}**")
    
    if st.button("Transcribe", type="primary"):
        if not url:
            st.error("Please enter a YouTube URL")
        else:
            with st.spinner("Downloading audio..."):
                try:
                    # Download audio using yt-dlp
                    audio_path = TEMP_FOLDER / f"audio.{SUPPORTED_AUDIO_FORMAT}"
                    
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': str(audio_path.with_suffix('')),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': SUPPORTED_AUDIO_FORMAT,
                            'preferredquality': '192',
                        }],
                    }
                    
                    video_slug = "transcription"
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        title = info.get("title") if info else None
                        video_id = info.get("id") if info else None
                        if title or video_id:
                            label = f"{title or ''}_{video_id or ''}".strip("_")
                        else:
                            label = url
                        video_slug = slugify_filename(label, "transcription")
                        ydl.download([url])
                    
                    # Find the downloaded file
                    audio_file = None
                    expected_file = TEMP_FOLDER / f"audio.{SUPPORTED_AUDIO_FORMAT}"
                    if expected_file.exists():
                        audio_file = expected_file
                    else:
                        for ext in ['.m4a', '.webm', '.mp3', '.opus']:
                            potential_file = TEMP_FOLDER / f"audio{ext}"
                            if potential_file.exists():
                                audio_file = potential_file
                                break
                    
                    if not audio_file or not audio_file.exists():
                        st.error("Failed to download audio file")
                        st.info(f"Checked for files in: {TEMP_FOLDER}")
                        st.stop()
                    
                    st.success("Audio downloaded successfully!")
                    
                    # Determine language for transcription
                    language_code = None
                    if language_option == "English (SA Accent)":
                        language_code = "en"
                    elif language_option == "Thai":
                        language_code = "th"
                    
                    with st.spinner("Transcribing audio (this may take a while)..."):
                        # Load Whisper model (CACHED - won't reload if already loaded)
                        model = load_whisper_model(model_size, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
                        
                        # Transcribe with VAD filter for ~30% speed improvement
                        segments, info = model.transcribe(
                            str(audio_file),
                            language=language_code,
                            task="transcribe",
                            vad_filter=True,  # Skip silent parts - major speed boost
                            beam_size=beam_size
                        )
                        
                        # Collect transcription text
                        full_text = []
                        srt_content = []
                        srt_index = 1
                        
                        for segment in segments:
                            text = segment.text.strip()
                            full_text.append(text)
                            
                            # SRT format
                            start_time = format_timestamp(segment.start)
                            end_time = format_timestamp(segment.end)
                            srt_content.append(f"{srt_index}\n{start_time} --> {end_time}\n{text}\n")
                            srt_index += 1
                        
                        transcription_text = " ".join(full_text)
                        
                        # Save files
                        txt_filename = f"{video_slug}.txt"
                        srt_filename = f"{video_slug}.srt"
                        txt_path = TEMP_FOLDER / txt_filename
                        srt_path = TEMP_FOLDER / srt_filename
                        
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(transcription_text)
                        
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(srt_content))
                        
                        st.success("Transcription complete!")
                        
                        # Display preview
                        st.subheader("Preview")
                        st.text_area("Transcription Preview", transcription_text, height=200)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            with open(txt_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download .txt",
                                    data=f.read(),
                                    file_name=txt_filename,
                                    mime="text/plain"
                                )
                        
                        with col2:
                            with open(srt_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download .srt",
                                    data=f.read(),
                                    file_name=srt_filename,
                                    mime="text/plain"
                                )
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)

elif tool == "The Digitizer (PDF OCR)":
    st.title("📄 The Digitizer - PDF OCR")
    st.markdown("Make your PDFs searchable with OCR technology. Upload multiple files for batch processing.")
    
    # Check for Tesseract
    if not check_tesseract():
        st.warning("⚠️ Please install Tesseract on your computer first.")
        st.info("""
        **Installation Instructions:**
        - **Windows:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
        - **macOS:** `brew install tesseract`
        - **Linux:** `sudo apt-get install tesseract-ocr`
        """)
        st.stop()
    
    st.info(f"📊 OCR will use **{OCR_JOBS} parallel jobs** (optimized for your {CPU_LOGICAL_PROCESSORS} threads)")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or Image Files (Multiple files supported)", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    # Initialize queue if needed
    if "ocr_file_queue" not in st.session_state:
        st.session_state["ocr_file_queue"] = []
    if "ocr_processed_files" not in st.session_state:
        st.session_state["ocr_processed_files"] = []
    
    # Add new files to queue
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if file already in queue
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            existing = [f for f in st.session_state["ocr_file_queue"] if f.get("id") == file_id]
            existing_processed = [f for f in st.session_state["ocr_processed_files"] if f.get("id") == file_id]
            
            if not existing and not existing_processed:
                # Save file to temp
                file_ext = Path(uploaded_file.name).suffix.lower()
                file_type = "pdf" if file_ext == ".pdf" else "image"
                file_path = TEMP_FOLDER / f"ocr_input_{len(st.session_state['ocr_file_queue'])}{file_ext}"
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state["ocr_file_queue"].append({
                    "id": file_id,
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "type": file_type,
                    "path": str(file_path),
                    "status": "pending"
                })
    
    # Display file queue
    if st.session_state["ocr_file_queue"] or st.session_state["ocr_processed_files"]:
        st.subheader("📋 File Queue")
        
        # Show pending files
        if st.session_state["ocr_file_queue"]:
            st.markdown("**Pending Files:**")
            queue_df_data = []
            for file_info in st.session_state["ocr_file_queue"]:
                queue_df_data.append({
                    "Filename": file_info["name"],
                    "Size": format_file_size(file_info["size"]),
                    "Type": file_info["type"].upper(),
                    "Status": "⏳ Pending"
                })
            
            queue_df = pd.DataFrame(queue_df_data)
            st.dataframe(queue_df, use_container_width=True, hide_index=True)
        
        # Show processed files
        if st.session_state["ocr_processed_files"]:
            st.markdown("**Processed Files:**")
            processed_df_data = []
            for file_info in st.session_state["ocr_processed_files"]:
                status_icon = "✅" if file_info["status"] == "completed" else "❌"
                processed_df_data.append({
                    "Filename": file_info["name"],
                    "Size": format_file_size(file_info["size"]),
                    "Status": f"{status_icon} {file_info['status'].title()}"
                })
            
            processed_df = pd.DataFrame(processed_df_data)
            st.dataframe(processed_df, use_container_width=True, hide_index=True)
            
            # Download buttons for completed files
            st.markdown("**Download Processed Files:**")
            for file_info in st.session_state["ocr_processed_files"]:
                if file_info["status"] == "completed" and file_info.get("output_path"):
                    output_path = Path(file_info["output_path"])
                    if output_path.exists():
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Download {file_info['output_filename']}",
                                data=f.read(),
                                file_name=file_info["output_filename"],
                                mime="application/pdf",
                                key=f"download_{file_info['id']}"
                            )
        
        # Process button
        if st.session_state["ocr_file_queue"]:
            if st.button("🚀 Process All Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_files = len(st.session_state["ocr_file_queue"])
                processed_count = 0
                
                # Process files sequentially
                queue_copy = st.session_state["ocr_file_queue"].copy()
                for idx, file_info in enumerate(queue_copy):
                    status_text.text(f"Processing {file_info['name']} ({idx + 1}/{total_files})...")
                    
                    # Update status to processing
                    file_info["status"] = "processing"
                    
                    # Process the file
                    result = process_single_ocr_file(file_info)
                    
                    # Move to processed list
                    file_info["status"] = result["status"]
                    file_info["output_path"] = result.get("output_path")
                    file_info["output_filename"] = result.get("output_filename")
                    file_info["error"] = result.get("error")
                    
                    st.session_state["ocr_processed_files"].append(file_info)
                    st.session_state["ocr_file_queue"].remove(
                        next(f for f in st.session_state["ocr_file_queue"] if f["id"] == file_info["id"])
                    )
                    
                    processed_count += 1
                    progress_bar.progress(processed_count / total_files)
                    
                    if result["status"] == "error":
                        st.warning(f"Error processing {file_info['name']}: {result['error']}")
                
                status_text.text("✅ Batch processing complete!")
                st.success(f"Processed {processed_count} file(s). Files are saved and ready for download.")
                st.rerun()
        
        # Clear queue button
        if st.session_state["ocr_file_queue"]:
            if st.button("🗑️ Clear Pending Queue"):
                # Remove files from disk
                for file_info in st.session_state["ocr_file_queue"]:
                    file_path = Path(file_info["path"])
                    if file_path.exists():
                        file_path.unlink()
                st.session_state["ocr_file_queue"] = []
                st.rerun()
        
        # Clear processed files button
        if st.session_state["ocr_processed_files"]:
            if st.button("🗑️ Clear Processed Files"):
                # Remove output files from disk
                for file_info in st.session_state["ocr_processed_files"]:
                    if file_info.get("output_path"):
                        output_path = Path(file_info["output_path"])
                        if output_path.exists():
                            output_path.unlink()
                st.session_state["ocr_processed_files"] = []
                st.rerun()

elif tool == "The Auto-Marker (Memo Generator)":
    st.title("✅ The Auto-Marker - Memo Generator")
    st.markdown("Generate marking memos from PDF pages or images using Gemini. Use the chat interface to correct AI assumptions.")
    
    if not gemini_status:
        st.error("⚠️ Gemini API key not configured")
        st.info("""
        **To enable this tool, set up your API key:**
        
        1. Create a `.env` file in the project root directory
        2. Add the following line:
           ```
           GEMINI_API_KEY=your_api_key_here
           ```
        3. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        4. Restart the Streamlit app
        
        The API key must be set in the `.env` file - manual entry is not supported.
        """)
    
    # Initialize chat and context if needed
    if "memo_chat_history" not in st.session_state:
        st.session_state["memo_chat_history"] = []
    if "memo_context" not in st.session_state:
        st.session_state["memo_context"] = {
            "ocr_results": {},
            "extracted_passages": {},
            "generated_memo": "",
            "page_numbers": [],
            "page_mapping": {},
            "pdf_path": None
        }
    
    # Main content area - split into two columns for chat
    col_main, col_chat = st.columns([2, 1])
    
    with col_main:
        uploaded_file = st.file_uploader(
            "Upload PDF, PNG, or JPG", 
            type=["pdf", "png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            file_ext = Path(uploaded_file.name).suffix.lower()
            source_name = Path(uploaded_file.name).stem
            st.session_state["memo_source_name"] = source_name
            
            if file_ext == ".pdf":
                st.subheader("PDF Page Selection")
                
                pdf_path = TEMP_FOLDER / "memo_input.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Store PDF path in context
                st.session_state["memo_context"]["pdf_path"] = str(pdf_path)
                
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()
                
                if not st.session_state["memo_page_input"]:
                    st.session_state["memo_page_input"] = "1"
                
                page_input = st.text_input(
                    "Pages to Process (e.g., 10-15, 22, 50-60)",
                    value=st.session_state["memo_page_input"],
                    placeholder="1-3, 5, 10-12"
                )
                if page_input != st.session_state["memo_page_input"]:
                    st.session_state["memo_page_input"] = page_input
                
                st.caption("Tip: Keep batches small (3-5 pages) for best performance.")
                
                if st.button("Load Pages", type="primary"):
                    parsed_pages, errors = parse_page_input(page_input, total_pages)
                    if errors:
                        for err in errors:
                            st.warning(err)
                    if not parsed_pages:
                        st.warning("No valid pages to load. Please adjust your input.")
                    else:
                        st.session_state["memo_pages_label"] = compress_pages(parsed_pages)
                        st.session_state["memo_context"]["page_numbers"] = parsed_pages.copy()
                        
                        # Extract OCR text for context
                        ocr_results = {}
                        for page_num in parsed_pages:
                            ocr_text = extract_text_from_pdf_page(pdf_path, page_num)
                            ocr_results[page_num] = ocr_text
                        st.session_state["memo_context"]["ocr_results"] = ocr_results
                        
                        new_images = []
                        with st.spinner("Converting selected pages to images..."):
                            try:
                                doc = fitz.open(pdf_path)
                                try:
                                    for page_number in parsed_pages:
                                        page = doc[page_number - 1]
                                        mat = fitz.Matrix(2.0, 2.0)
                                        pix = page.get_pixmap(matrix=mat)
                                        image_path = TEMP_FOLDER / f"page_{page_number}.png"
                                        pix.save(str(image_path))
                                        new_images.append({
                                            "path": str(image_path),
                                            "label": f"{uploaded_file.name} • Page {page_number}",
                                            "size": (pix.width, pix.height),
                                            "pdf_page": page_number
                                        })
                                finally:
                                    doc.close()
                                
                                if new_images:
                                    st.session_state["memo_images"] = new_images
                                    # Initialize page mapping (PDF page = displayed page by default)
                                    page_mapping = {img["pdf_page"]: img["pdf_page"] for img in new_images}
                                    st.session_state["memo_context"]["page_mapping"] = page_mapping
                                    st.success(f"Loaded {len(new_images)} page(s) successfully.")
                                else:
                                    st.warning("No pages were loaded. Please adjust your selection and try again.")
                            except Exception as e:
                                st.error(f"Error processing PDF: {str(e)}")
                                st.exception(e)
            
            else:
                image_path = TEMP_FOLDER / uploaded_file.name
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with Image.open(image_path) as img:
                    size = img.size
                
                st.session_state["memo_pages_label"] = "Image"
                st.session_state["memo_page_input"] = ""
                st.session_state["memo_images"] = [{
                    "path": str(image_path),
                    "label": uploaded_file.name,
                    "size": size,
                    "pdf_page": 1
                }]
                st.session_state["memo_context"]["page_numbers"] = [1]
                st.session_state["memo_context"]["page_mapping"] = {1: 1}
                st.success("Image loaded successfully!")
        
        memo_images = st.session_state.get("memo_images", [])
        
        if memo_images:
            st.subheader("Loaded Pages")
            for idx, img_meta in enumerate(memo_images, start=1):
                pdf_page = img_meta.get("pdf_page", idx)
                st.write(f"**{idx}. {img_meta['label']}** — {img_meta['size'][0]}x{img_meta['size'][1]} px (PDF Page {pdf_page})")
            
            preview_path = Path(memo_images[0]["path"])
            if preview_path.exists():
                with Image.open(preview_path) as preview_img:
                    st.image(preview_img, caption=memo_images[0]["label"], width="stretch")
            else:
                st.warning("Preview image missing from disk. Reload the pages to continue.")
        
        if memo_images:
            if st.button("Generate Memo", type="primary"):
                if not gemini_status:
                    st.error("⚠️ Gemini API key not configured. Set GEMINI_API_KEY in your .env file and restart the app.")
                else:
                    with st.spinner("Gemini is generating memos..."):
                        try:
                            model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=SAFETY_SETTINGS)
                            memo_sections = []
                            
                            for idx, img_meta in enumerate(memo_images, start=1):
                                image_path = Path(img_meta["path"])
                                if not image_path.exists():
                                    st.warning(f"Skipped {img_meta['label']} (file missing).")
                                    continue
                                
                                current_size = img_meta.get("size")
                                if current_size and max(current_size) > MAX_IMAGE_SIZE:
                                    image_path = resize_image_for_model(image_path, MAX_IMAGE_SIZE)
                                
                                with Image.open(image_path) as pil_image:
                                    system_prompt = (
                                        "You are a Grade 3 Teacher. Analyze the worksheet page and:\n"
                                        "1. Identify each question.\n"
                                        "2. Solve it accurately.\n"
                                        "3. Produce a clean marking memo with numbered answers.\n"
                                        "If the page has no questions, respond with 'No questions found.'"
                                    )
                                    
                                    response = model.generate_content(
                                        [system_prompt, pil_image],
                                        safety_settings=SAFETY_SETTINGS
                                    )
                                    memo_text = (response.text or "").strip()
                                
                                if not memo_text:
                                    memo_text = "No response generated."
                                
                                memo_sections.append(f"### {img_meta['label']}\n{memo_text}")
                                
                                # Store extracted passages in context
                                pdf_page = img_meta.get("pdf_page", idx)
                                st.session_state["memo_context"]["extracted_passages"][pdf_page] = memo_text
                                
                                if idx < len(memo_images):
                                    st.info(f"Rate Limit Safety: Waiting {GEMINI_DELAY_SECONDS} seconds before starting the next page...")
                                    time.sleep(GEMINI_DELAY_SECONDS)
                            
                            if memo_sections:
                                combined_memo = "\n\n".join(memo_sections)
                                st.session_state["memo_context"]["generated_memo"] = combined_memo
                                st.success("Memo generated!")
                                st.subheader("Generated Memo")
                                st.markdown(combined_memo)
                                
                                source_name = st.session_state.get("memo_source_name") or "Memo"
                                pages_label = st.session_state.get("memo_pages_label") or compress_pages(
                                    list(range(1, len(memo_images) + 1))
                                )
                                memo_filename = build_memo_filename(source_name, pages_label)
                                
                                memo_path = TEMP_FOLDER / memo_filename
                                with open(memo_path, "w", encoding="utf-8") as f:
                                    f.write(combined_memo)
                                
                                with open(memo_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Memo",
                                        data=f.read(),
                                        file_name=memo_filename,
                                        mime="text/plain"
                                    )
                            else:
                                st.warning("No memos were generated. Please retry.")
                        
                        except Exception as e:
                            st.error(f"Error generating memo: {str(e)}")
                            st.exception(e)
        
        # Always display the current memo if it exists (from initial generation or regeneration)
        if st.session_state["memo_context"].get("generated_memo"):
            st.markdown("---")
            st.subheader("📄 Generated Memo")
            st.markdown(st.session_state["memo_context"]["generated_memo"])
            
            # Download button for current memo
            source_name = st.session_state.get("memo_source_name") or "Memo"
            pages_label = st.session_state.get("memo_pages_label") or compress_pages(
                list(range(1, len(memo_images) + 1)) if memo_images else []
            )
            memo_filename = build_memo_filename(source_name, pages_label)
            memo_path = TEMP_FOLDER / memo_filename
            
            if memo_path.exists():
                with open(memo_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Memo",
                        data=f.read(),
                        file_name=memo_filename,
                        mime="text/plain",
                        key="download_memo_main"
                    )
    
    # Chat interface in sidebar column
    with col_chat:
        st.subheader("💬 Correction Chat")
        st.caption("Correct AI assumptions about page references and content.")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state["memo_chat_history"]:
                for msg in st.session_state["memo_chat_history"]:
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.write(msg["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(msg["content"])
            else:
                st.info("💡 Use this chat to correct the AI's assumptions. For example:\n\n"
                       "- \"The questions for this passage are actually on page 17 (top right corner).\"\n"
                       "- \"Use the page number printed on the bottom left instead of the PDF page number.\"\n"
                       "- \"This passage belongs to the section before the table, not after.\"")
        
        # Chat input
        if memo_images and st.session_state["memo_context"].get("generated_memo"):
            user_input = st.text_input(
                "Type your correction or question:",
                key="chat_input",
                placeholder="e.g., Questions are on page 17, not 15..."
            )
            
            if st.button("Send", key="send_chat") and user_input:
                # Add user message to history
                st.session_state["memo_chat_history"].append({
                    "role": "user",
                    "content": user_input
                })
                
                # Process correction with AI
                with st.spinner("Processing correction..."):
                    try:
                        model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=SAFETY_SETTINGS)
                        
                        # Build context for AI
                        context_prompt = f"""You are helping correct a memo generation system. The user has provided a correction.

CONTEXT:
- OCR Results Available: {list(st.session_state['memo_context']['ocr_results'].keys())}
- Page Numbers Processed: {st.session_state['memo_context']['page_numbers']}
- Current Page Mapping: {st.session_state['memo_context']['page_mapping']}
- Generated Memo: {st.session_state['memo_context']['generated_memo'][:500]}...

USER CORRECTION: {user_input}

Please:
1. Acknowledge the correction
2. Explain what will be updated
3. If page numbers need to change, specify which pages
4. Ask if the user wants to regenerate the memo with these corrections

Be concise and helpful."""

                        response = model.generate_content(
                            context_prompt,
                            safety_settings=SAFETY_SETTINGS
                        )
                        ai_response = (response.text or "").strip()
                        
                        # Add AI response to history
                        st.session_state["memo_chat_history"].append({
                            "role": "assistant",
                            "content": ai_response
                        })
                        
                        # Try to extract page number corrections from user input
                        page_num_pattern = r'page\s+(\d+)'
                        matches = re.findall(page_num_pattern, user_input.lower())
                        if matches:
                            # Update page mapping if page numbers mentioned
                            for match in matches:
                                page_num = int(match)
                                # Try to infer which page should be updated
                                # This is a simple heuristic - can be improved
                                if page_num not in st.session_state["memo_context"]["page_mapping"]:
                                    # Add new page mapping
                                    st.session_state["memo_context"]["page_mapping"][page_num] = page_num
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error processing correction: {str(e)}")
                        st.session_state["memo_chat_history"].append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })
                        st.rerun()
            
            # Regenerate memo button
            if st.session_state["memo_chat_history"]:
                if st.button("🔄 Regenerate Memo with Corrections", type="primary"):
                    if not gemini_status:
                        st.error("⚠️ Gemini API key not configured. Set GEMINI_API_KEY in your .env file and restart the app.")
                    else:
                        with st.spinner("Regenerating memo with corrections..."):
                            try:
                                model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=SAFETY_SETTINGS)
                                
                                # Build correction context
                                correction_summary = "\n".join([
                                    f"{msg['role']}: {msg['content']}" 
                                    for msg in st.session_state["memo_chat_history"]
                                ])
                                
                                memo_sections = []
                                for idx, img_meta in enumerate(memo_images, start=1):
                                    image_path = Path(img_meta["path"])
                                    if not image_path.exists():
                                        continue
                                    
                                    current_size = img_meta.get("size")
                                    if current_size and max(current_size) > MAX_IMAGE_SIZE:
                                        image_path = resize_image_for_model(image_path, MAX_IMAGE_SIZE)
                                    
                                    with Image.open(image_path) as pil_image:
                                        system_prompt = (
                                            f"You are a Grade 3 Teacher. Analyze the worksheet page and:\n"
                                            f"1. Identify each question.\n"
                                            f"2. Solve it accurately.\n"
                                            f"3. Produce a clean marking memo with numbered answers.\n"
                                            f"\nIMPORTANT CORRECTIONS FROM USER:\n{correction_summary}\n"
                                            f"Please apply these corrections when analyzing this page.\n"
                                            f"If the page has no questions, respond with 'No questions found.'"
                                        )
                                        
                                        response = model.generate_content(
                                            [system_prompt, pil_image],
                                            safety_settings=SAFETY_SETTINGS
                                        )
                                        memo_text = (response.text or "").strip()
                                    
                                    if not memo_text:
                                        memo_text = "No response generated."
                                    
                                    memo_sections.append(f"### {img_meta['label']}\n{memo_text}")
                                    
                                    pdf_page = img_meta.get("pdf_page", idx)
                                    st.session_state["memo_context"]["extracted_passages"][pdf_page] = memo_text
                                    
                                    if idx < len(memo_images):
                                        time.sleep(GEMINI_DELAY_SECONDS)
                                
                                if memo_sections:
                                    combined_memo = "\n\n".join(memo_sections)
                                    st.session_state["memo_context"]["generated_memo"] = combined_memo
                                    
                                    # Save to file
                                    source_name = st.session_state.get("memo_source_name") or "Memo"
                                    pages_label = st.session_state.get("memo_pages_label") or compress_pages(
                                        list(range(1, len(memo_images) + 1))
                                    )
                                    memo_filename = build_memo_filename(source_name, pages_label)
                                    memo_path = TEMP_FOLDER / memo_filename
                                    with open(memo_path, "w", encoding="utf-8") as f:
                                        f.write(combined_memo)
                                    
                                    # Add confirmation to chat
                                    st.session_state["memo_chat_history"].append({
                                        "role": "assistant",
                                        "content": "✅ Memo regenerated with your corrections applied!"
                                    })
                                    
                                    st.success("Memo regenerated!")
                                    st.rerun()
                            
                            except Exception as e:
                                st.error(f"Error regenerating memo: {str(e)}")
                                st.exception(e)
            
            # Clear chat button
            if st.session_state["memo_chat_history"]:
                if st.button("🗑️ Clear Chat"):
                    st.session_state["memo_chat_history"] = []
                    st.rerun()
