<<<<<<< HEAD
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
- AI-powered analysis using Google Gemini
- Interactive chat interface for correcting AI assumptions
- Page selection for PDFs with preview
- Batch OCR processing for multiple files

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
4. **Google Gemini API Key** (for AI memo generation)
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - See "API Key Setup" section below for configuration options

### Setup

1. Clone or download this repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. **API Key Setup** (for Auto-Marker tool):
   
   You have three options to set your Gemini API key:
   
   **Option 1: Environment Variable (Recommended)**
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Windows (Command Prompt)
   set GEMINI_API_KEY=your_api_key_here
   
   # macOS/Linux
   export GEMINI_API_KEY="your_api_key_here"
   ```
   
   **Option 2: .env File**
   ```bash
   # Create a .env file in the project root
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```
   Note: Requires `python-dotenv` package (included in requirements.txt)
   
   **Option 3: Manual Entry**
   - If no API key is set, you can enter it in the sidebar when using the app
   - This must be done each time you restart the app

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. **The Archive**: Enter a YouTube URL, select language and model size, then click "Transcribe"
2. **The Digitizer**: 
   - Upload multiple PDF or image files for batch processing
   - Files are processed sequentially and saved as they complete
   - Download processed files individually
3. **The Auto-Marker**: 
   - Upload a PDF or image, select pages (if PDF), preview, then generate the memo
   - Use the chat interface to correct AI assumptions about page references
   - Regenerate memos with corrections applied

## Notes

- All temporary files are stored in `temp_files/` folder and cleaned on startup
- The application runs completely offline (except for YouTube downloads and Gemini API calls)
- Bandwidth-efficient: only downloads audio, not video
- API key is stored securely: environment variables are preferred over manual entry
- Batch OCR processing saves files as they complete (abort-safe)

## Troubleshooting

- **Tesseract not found**: Install Tesseract and ensure it's in your system PATH
- **Gemini API key not working**: 
  - Check that your API key is set correctly (environment variable or .env file)
  - Verify the key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
  - Check the sidebar for connection status
- **FFmpeg errors**: Install FFmpeg and ensure it's in your system PATH
- **Model download issues**: Some models may need to be downloaded on first use
- **Batch OCR processing stopped**: Completed files are saved, you can download them even if processing was interrupted

=======
# ocr-pdf-transcribe
Wraps up 3 tasks in App - OCR PDFs, Transcribe Videos, Memo Generator
>>>>>>> main
