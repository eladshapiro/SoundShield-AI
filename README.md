# Kindergarten Recording Analyzer

Advanced system for analyzing kindergarten audio recordings to detect inappropriate staff behavior.

## Features

- **Emotion Detection** - Detects anger, stress, and aggression in staff voices
- **Baby Cry Detection** - Detects crying and checks staff response
- **Neglect Detection** - Detects lack of response to crying and lack of interaction
- **Violence Detection** - Detects shouting, threats, and verbal violence
- **Detailed Reports** - Comprehensive reports with timestamps
- **Modern GUI** - User-friendly interface with language selection
- **Web Interface** - Web-based interface for file upload and results viewing

## Accuracy on real audio (v3.0)

v3.0 replaced the hand-tuned spectral heuristics with pretrained audio models as
the primary detectors and added a real-audio evaluation suite (`evaluation/`,
~7,000 labelled public clips: ESC-50, Donate-a-Cry, RAVDESS, CREMA-D, VIVAE,
LibriSpeech, FLEURS).  Thresholds were tuned on a dev split; the numbers below
are on the held-out test split, comparing the v2.5 heuristics with the v3.0
production rules.

| task | v2.5 heuristics (P / R / F1) | v3.0 models (P / R / F1) | false-positive rate v2.5 → v3.0 |
|---|---|---|---|
| infant cry present | 0.19 / 0.92 / 0.31 | **0.97 / 0.88 / 0.92** | 76% → 0.5% |
| vocal aggression (screams, strong anger) | 0.08 / 0.17 / 0.11 | **0.74 / 0.79 / 0.77** | 20% → 2.8% |
| adult speech present (staff-response signal) | 0.00 / 0.00 / 0.00 | **0.97 / 0.99 / 0.98** | 0% → 4.4% |
| angry adult speech | 0.80 / 0.09 / 0.15 | **0.63 / 0.66 / 0.64** | 0.7% → 13% |

| transcription | Whisper `base` (v2.5) | v3.0 model |
|---|---|---|
| Hebrew (FLEURS he_il, WER) | 66.1% | **21.4%** (`ivrit-ai/whisper-large-v3-turbo-ct2`) |
| English (LibriSpeech dev-clean, WER) | 5.8% | **4.4%** (`large-v3-turbo`) |

Anger is the hardest task: acted, exuberant "happy" speech (RAVDESS strong
intensity) still triggers the arousal/dominance rule about 60% of the time,
so `aggression` flags from the dimensional model alone are reported one severity
level below `anger` flags that both emotion models agree on.  On 20-minute
montages of held-out clips the full pipeline caught 15/15 inserted cries and
11/17 screams at ~17× realtime on an RTX 4070 Ti; see
`evaluation/results/` and `evaluation/README.md` for the full breakdown and
the false-alarm-per-hour figures.

Models (downloaded from Hugging Face on first use): `MIT/ast-finetuned-audioset-10-10-0.4593`
(event tagger), `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (arousal /
dominance / valence), `superb/hubert-large-superb-er` (categorical emotion),
faster-whisper `large-v3-turbo` / `ivrit-ai/whisper-large-v3-turbo-ct2`.
Everything runs on CPU if no GPU is present (set `WHISPER_MODEL_EN=base` and
`TAGGER_HOP_SECONDS=3` there for speed); the heuristics remain as the fallback
when a model cannot be loaded.

## Language Support

The application supports two languages:
- **English** - Full interface and analysis support
- **Hebrew** - Full interface and analysis support

You can select your preferred language in the GUI interface. The selected language will be used for:
- Interface text
- Audio transcription (Whisper)
- Inappropriate language detection
- Report generation

## Installation

### 1. Install System Dependencies:

> **📖 For detailed FFmpeg installation instructions, see [INSTALL_FFMPEG.md](INSTALL_FFMPEG.md)**

#### Windows - Install FFmpeg (Required for Whisper transcription):

**Option A: Using Chocolatey (Recommended)**
```powershell
# Install Chocolatey first (if not installed):
# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then install ffmpeg:
choco install ffmpeg
```

**Option B: Manual Installation**
1. Download FFmpeg from: https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH:
   - Open System Properties → Environment Variables
   - Add `C:\ffmpeg\bin` to PATH
   - Restart terminal/PowerShell

**Option C: Using winget (Windows 10/11)**
```powershell
winget install ffmpeg
```

**Verify Installation:**
```bash
# IMPORTANT: Restart your terminal/PowerShell after installation!
ffmpeg -version
```

> ⚠️ **Important:** After installing FFmpeg, you MUST restart your terminal/PowerShell for the PATH changes to take effect!

#### Linux:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

### 2. Install Python Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Create Required Directories:
```bash
mkdir uploads reports models
```

## Usage

### 1. GUI Interface (Recommended):
```bash
python gui_app.py
```
or
```bash
python run_gui.py
```

The GUI provides:
- 📁 Easy file selection
- 🌐 Language selection (English/Hebrew)
- 📊 Real-time progress tracking
- 📋 Detailed results display
- 🎨 Modern, user-friendly interface

**Language Selection:**
- Select your preferred language at the top of the GUI
- The selected language will be used for audio analysis
- Interface text will update immediately

### 2. Command Line Usage:
```bash
# Analyze a single file
python main.py recording.wav

# Run example
python example_usage.py
```

### 3. Web Interface:
```bash
python web_app.py
```
Then open in browser: http://localhost:5000

### 4. Programmatic Usage:
```python
from main import KindergartenRecordingAnalyzer

# Initialize with language (default: 'en')
analyzer = KindergartenRecordingAnalyzer(language='en')  # or 'he' for Hebrew

# Run complete analysis
results = analyzer.run_complete_analysis("recording.wav")
report = analyzer.generate_report(results)
```

## Supported Audio Formats

- WAV, MP3, M4A, FLAC, AAC, OGG
- Up to 500MB per file
- All sample rates supported (recommended 22kHz+)

## File Structure

### Core Modules:
- `main.py` - Main application entry point
- `gui_app.py` - Modern GUI application with language support
- `audio_analyzer.py` - Basic audio analysis module
- `emotion_detector.py` - Emotion and speech tone detection
- `cry_detector.py` - Baby cry detection and response checking
- `violence_detector.py` - Violence and aggression detection
- `neglect_detector.py` - Neglect and lack of attention detection
- `report_generator.py` - Detailed report generation
- `advanced_analyzer.py` - Advanced ML models (Whisper, Transformers)
- `inappropriate_language_detector.py` - Profanity and inappropriate language detection

### Web Interface:
- `web_app.py` - Flask web application
- `templates/index.html` - User interface (auto-generated)

### Examples & Documentation:
- `example_usage.py` - Detailed usage example
- `requirements.txt` - Python dependencies list
- `README.md` - This guide

### Generated Directories:
- `uploads/` - Uploaded audio files
- `reports/` - Generated reports (JSON, HTML, CSV)
- `models/` - Trained models (if any)

## Analysis Output

The system generates reports in different formats:

### 1. JSON Report:
- Structured data for further analysis
- Includes all raw results

### 2. HTML Report:
- Visual report
- Includes graphs and summaries
- Viewable in browser

### 3. CSV Report:
- Table of all events
- Suitable for import to Excel or other analysis tools

## Key Metrics

### Emotion Analysis:
- **Anger** - Detects anger in staff voices
- **Stress** - Detects tension and stress
- **Aggression** - Detects aggressive behavior
- **Calm** - Detects calm and normal tone

### Cry Detection:
- **Cry Detection** - Automatic detection of baby crying
- **Cry Intensity** - Weak, medium, strong
- **Staff Response** - Checks if staff responded to crying
- **Response Quality** - Evaluates response quality

### Violence Detection:
- **Shouting** - Detects shouting and aggression
- **Aggressive Tone** - Detects threatening tone
- **Physical Violence** - Signs of potential physical violence
- **Context** - Analysis of context around events

### Neglect Detection:
- **Unanswered Cries** - Detects crying without response
- **Prolonged Silence** - Detects unusual quiet periods
- **Lack of Interaction** - Detects lack of communication with children
- **Distress Neglect** - Detects ignoring children's distress

## Severity Levels

- **Low** - Minor events requiring monitoring
- **Medium** - Events requiring attention
- **High** - Events requiring immediate action
- **Critical** - Events requiring urgent intervention

## Privacy & Ethics

⚠️ **Very Important:**
- The system is intended for child protection only
- Must be used in accordance with local law
- Explicit consent must be obtained before use
- Data is sensitive and requires maximum protection
- Consult with a legal expert before use

## Technical Requirements

### Minimum:
- Python 3.8+
- 4GB RAM
- 1GB free space
- **FFmpeg** (required for Whisper transcription)

### Recommended:
- Python 3.9+
- 8GB+ RAM
- SSD with 5GB+ free space
- Fast processor (i5 and above)
- **FFmpeg** installed and in PATH

### System Dependencies:
- **FFmpeg** - Required for audio transcription with Whisper
  - Windows: Install via Chocolatey or download from https://ffmpeg.org/download.html
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

## Troubleshooting

### Common Errors:

1. **"ModuleNotFoundError"**
   ```bash
   pip install -r requirements.txt
   ```

2. **"Audio file not found"**
   - Verify the file path is correct
   - Check that the file exists and is readable

3. **"Out of memory"**
   - Try a shorter audio file
   - Close other applications

4. **"Web interface not loading"**
   - Check that port 5000 is available
   - Try: http://127.0.0.1:5000

5. **"Language detection issues"**
   - Make sure you selected the correct language in the GUI
   - The audio file should match the selected language for best results

6. **"Whisper transcription failed" / "ffmpeg not found"**
   - **This is the most common issue!**
   - FFmpeg is required for Whisper to work
   - Install FFmpeg using one of the methods in the Installation section
   - After installation, restart your terminal/PowerShell
   - Verify with: `ffmpeg -version`
   - **Note:** The analysis will continue without transcription, but language detection won't work

## Support

For questions and technical support:
- Check this guide first
- Run `python example_usage.py` for testing
- Ensure all dependencies are installed

## License

The system is intended for educational and research use only.
Must be used in accordance with local law and ethical principles.

---

**Important Note:** This system is a tool only and is not a substitute for professional human judgment. All decisions regarding child care should be made by qualified professionals.
