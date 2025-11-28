# Educational Toolkit - Streamlit Application

A local, offline educational toolkit for teachers in South Africa with three powerful tools for transcription, OCR, and AI-powered memo generation.

## Features

### 🎬 The Archive (YouTube Transcriber)
- Download audio from YouTube videos
- Transcribe to text and subtitle files (.txt and .srt)
- Support for English (SA Accent), Thai, and Auto-Detect
- Multiple model sizes (tiny, base, small, medium)

### 📄 The Digitizer (PDF OCR)
- Make PDFs searchable with OCR
- Automatic deskewing and optimization
- Smart text detection (skips existing text)

### ✅ The Auto-Marker (Memo Generator)
- Generate marking memos from PDF pages or images
- AI-powered analysis using Ollama (minicpm-v model)
- Page selection for PDFs with preview

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (for PDF OCR tool)
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`
3. **FFmpeg** (for YouTube audio download)
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`
4. **Ollama** (for AI memo generation)
   - Install from [ollama.ai](https://ollama.ai)
   - Pull the minicpm-v model: `ollama pull minicpm-v`
   - Start Ollama server: `ollama serve`

### Setup

1. Clone or download this repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start Ollama (if using Auto-Marker):
```bash
ollama serve
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. **The Archive**: Enter a YouTube URL, select language and model size, then click "Transcribe"
2. **The Digitizer**: Upload a PDF and click "Process PDF" to make it searchable
3. **The Auto-Marker**: Upload a PDF or image, select a page (if PDF), preview, then generate the memo

## Notes

- All temporary files are stored in `temp_files/` folder and cleaned on startup
- The application runs completely offline (except for YouTube downloads)
- Bandwidth-efficient: only downloads audio, not video
- Make sure Ollama is running before using the Auto-Marker tool

## Troubleshooting

- **Tesseract not found**: Install Tesseract and ensure it's in your system PATH
- **Ollama connection error**: Make sure `ollama serve` is running
- **FFmpeg errors**: Install FFmpeg and ensure it's in your system PATH
- **Model download issues**: Some models may need to be downloaded on first use

