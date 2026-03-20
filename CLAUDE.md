# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

SoundShield-AI analyzes kindergarten audio recordings to detect inappropriate staff behavior: emotional abuse, violence, neglect, inappropriate language, and inadequate response to child distress. Supports English and Hebrew.

## Commands

```bash
# Install
pip install -r requirements.txt
python install.py          # installs deps, creates dirs (uploads/ reports/ models/ templates/), validates

# Run
python main.py recording.wav              # CLI: analyze a single file
python gui_app.py                         # Desktop GUI (tkinter)
python web_app.py                         # Web UI at http://localhost:5000
python example_usage.py                   # Generate synthetic audio and run analysis demo

# Tests
python -m pytest tests/                   # Unit tests (unittest-based, in tests/)
python run_system_test.py                 # End-to-end test on synthetic audio
python run_5_tests.py                     # 5 comprehensive scenario tests (calm, shouting, cry, mixed, edge)
python -m pytest tests/test_main.py::TestKindergartenRecordingAnalyzer::test_initialization_valid_language  # single test
```

## Architecture

**Orchestrator pattern**: `main.py` contains `KindergartenRecordingAnalyzer` which initializes and coordinates all detector modules in a 7-step pipeline:

1. `AudioAnalyzer` (`audio_analyzer.py`) — load audio, extract features, segment into windows, **adaptive noise baseline** (first 30s calibration)
2. `EmotionDetector` (`emotion_detector.py`) — heuristic emotion detection; **HuBERT is PRIMARY** when available (merged via `merge_with_advanced_results()`)
3. `CryDetector` (`cry_detector.py`) — detect child crying, measure intensity/duration, verify staff response
4. `ViolenceDetector` (`violence_detector.py`) — detect shouting, threats, aggressive tone
5. `NeglectDetector` (`neglect_detector.py`) — detect unanswered crying, prolonged silence, lack of interaction
6. `AdvancedAnalyzer` (`advanced_analyzer.py`) — **eagerly loaded by default** (Whisper + HuBERT); `detect_concerning_emotions_advanced()` processes in 7s chunks
7. `InappropriateLanguageDetector` (`inappropriate_language_detector.py`) — Whisper transcription → word-list matching

**Additional modules:**
- `SpeakerDiarizer` (`speaker_diarizer.py`) — pyannote.audio or pitch-based fallback; classifies adult (<200Hz F0) vs child (>200Hz)
- `Database` (`database.py`) — SQLite persistence (analyses + incidents tables); powers history sidebar and comparison
- `ReportGenerator` (`report_generator.py`) — JSON/HTML/CSV reports with **Chart.js** visualizations, timeline bar, confidence-weighted severity scoring

ML models are the **default primary path** (`use_advanced=True`). Heuristics serve as fallback only when models aren't available. Each detection is tagged `ml_backed=True/False`.

**Web Frontend** (`web_app.py` + `templates/index.html` + `static/`):
- Modern dashboard: Tailwind CSS, Alpine.js, Chart.js, wavesurfer.js (all CDN, no build step)
- **SSE progress** via `/progress-stream/<filename>` (replaces polling)
- Waveform with colored incident overlays (violence=red, emotion=orange, cry=blue, neglect=gray, language=purple)
- Severity doughnut, incident type bar, timeline density, emotion radar charts
- Dark mode, RTL support (Hebrew), responsive mobile-first
- Modal reports (replaces `window.open` + `document.write`)
- Async analysis via `/upload-async` + `ThreadPoolExecutor`
- History sidebar, comparison endpoint (`/compare?ids=1,2,3`)

**Desktop GUI** (`gui_app.py`):
- Tabbed results: Summary | Charts (matplotlib) | Details
- EN/HE bilingual UI

Static files: `static/css/main.css`, `static/js/{app,upload,waveform,charts,modal}.js`

## Key Technical Details

- **Python 3.8+**, heavy deps: PyTorch, TensorFlow, Transformers, librosa, Whisper
- **FFmpeg required** for Whisper transcription (system dependency, not pip)
- Audio formats: WAV, MP3, M4A, FLAC, AAC, OGG; max 500MB; duration 1s–3600s
- All audio resampled to 16kHz via librosa
- Languages: `en` and `he` — passed to Whisper and word-list matching
- Windows console encoding fix applied in both `main.py` and `gui_app.py`
- Tests use `unittest` (not pytest fixtures); test files generate synthetic audio with numpy+soundfile
- No linter or formatter configured in the repo

## Commit Convention

```
<type>(<scope>): <subject>
```
Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `style`, `chore`
