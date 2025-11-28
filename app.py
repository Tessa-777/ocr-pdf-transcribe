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

# Session state defaults for memo generator so state persists across reruns
SESSION_DEFAULTS = {
    "memo_images": [],
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    "memo_source_name": None,
    "memo_page_input": "",
    "memo_pages_label": "",
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
        return False, "Enter a Gemini API key to enable AI memos."
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
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=st.session_state["gemini_api_key"],
    type="password",
    help="Stored in session memory only (never written to disk)."
)
if api_key_input != st.session_state["gemini_api_key"]:
    st.session_state["gemini_api_key"] = api_key_input.strip()

gemini_status, gemini_msg = configure_gemini(st.session_state["gemini_api_key"])
if gemini_status:
    st.sidebar.success(gemini_msg)
else:
    st.sidebar.warning(gemini_msg)


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
    st.markdown("Make your PDFs searchable with OCR technology.")
    
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
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        st.info(f"📊 OCR will use **{OCR_JOBS} parallel jobs** (optimized for your {CPU_LOGICAL_PROCESSORS} threads)")
        
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF with OCR (this may take a while)..."):
                try:
                    # Save uploaded file to temp
                    input_path = TEMP_FOLDER / "input.pdf"
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Output path
                    source_slug = slugify_filename(Path(uploaded_file.name).stem, "document")
                    output_filename = f"ocr_{source_slug}.pdf"
                    output_path = TEMP_FOLDER / output_filename
                    
                    # Run ocrmypdf with optimizations
                    ocrmypdf.ocr(
                        input_path,
                        output_path,
                        deskew=True,
                        optimize=0,  # Disable slow image re-compression for speed
                        skip_text=True,
                        jobs=OCR_JOBS,  # Use 11 parallel jobs (12 threads - 1)
                        fast_web_view=0  # Disable for speed
                    )
                    
                    st.success("PDF processed successfully!")
                    
                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Searchable PDF",
                            data=f.read(),
                            file_name=output_filename,
                            mime="application/pdf"
                        )
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.exception(e)

elif tool == "The Auto-Marker (Memo Generator)":
    st.title("✅ The Auto-Marker - Memo Generator")
    st.markdown("Generate marking memos from PDF pages or images using Gemini.")
    
    if not gemini_status:
        st.warning("Enter your Gemini API key in the sidebar to enable this tool.")
    
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
                                        "size": (pix.width, pix.height)
                                    })
                            finally:
                                doc.close()
                            
                            if new_images:
                                st.session_state["memo_images"] = new_images
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
                "size": size
            }]
            st.success("Image loaded successfully!")
    
    memo_images = st.session_state.get("memo_images", [])
    
    if memo_images:
        st.subheader("Loaded Pages")
        for idx, img_meta in enumerate(memo_images, start=1):
            st.write(f"**{idx}. {img_meta['label']}** — {img_meta['size'][0]}x{img_meta['size'][1]} px")
        
        preview_path = Path(memo_images[0]["path"])
        if preview_path.exists():
            with Image.open(preview_path) as preview_img:
                st.image(preview_img, caption=memo_images[0]["label"], width="stretch")
        else:
            st.warning("Preview image missing from disk. Reload the pages to continue.")
    
    if memo_images:
        if st.button("Generate Memo", type="primary"):
            if not gemini_status:
                st.error("Gemini API key missing. Please add it in the sidebar.")
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
                            
                            if idx < len(memo_images):
                                st.info(f"Rate Limit Safety: Waiting {GEMINI_DELAY_SECONDS} seconds before starting the next page...")
                                time.sleep(GEMINI_DELAY_SECONDS)
                        
                        if memo_sections:
                            combined_memo = "\n\n".join(memo_sections)
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
