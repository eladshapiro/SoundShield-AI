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
python web_app.py                         # Web UI at http://localhost:5000 (+ WebSocket on /ws)
python example_usage.py                   # Generate synthetic audio and run analysis demo
python benchmark.py                       # Performance benchmark for all detectors

# Tests
python -m pytest tests/                   # Full test suite (228 tests, unittest-based)
python -m pytest tests/test_api.py        # API endpoint smoke tests
python -m pytest tests/test_integration.py # Full pipeline integration tests
python run_system_test.py                 # End-to-end test on synthetic audio
python run_5_tests.py                     # 5 comprehensive scenario tests (calm, shouting, cry, mixed, edge)
python -m pytest tests/test_main.py::TestKindergartenRecordingAnalyzer::test_initialization_valid_language  # single test

# Docker
docker-compose up                         # Run with Docker (auto-builds)
docker build -t soundshield-ai .          # Build image only
```

## Architecture

**Orchestrator pattern**: `main.py` contains `KindergartenRecordingAnalyzer` which initializes and coordinates all detector modules in a 7-step pipeline:

1. `AudioAnalyzer` (`audio_analyzer.py`) — load audio, extract features, segment into windows, **adaptive noise baseline** (first 30s calibration)
2. `EmotionDetector` (`emotion_detector.py`) — heuristic emotion detection; **HuBERT is PRIMARY** when available (merged via `merge_with_advanced_results()`)
3. `CryDetector` (`cry_detector.py`) — detect child crying, measure intensity/duration, verify staff response, **response time metrics**, **escalation pattern detection**, **episode aggregation**
4. `ViolenceDetector` (`violence_detector.py`) — detect shouting, threats, aggressive tone
5. `NeglectDetector` (`neglect_detector.py`) — detect unanswered crying, prolonged silence, lack of interaction
6. `AdvancedAnalyzer` (`advanced_analyzer.py`) — **Faster-Whisper (priority) or OpenAI Whisper** + HuBERT (**ONNX Runtime priority, PyTorch fallback**); `detect_concerning_emotions_advanced()` processes in 7s chunks
7. `InappropriateLanguageDetector` (`inappropriate_language_detector.py`) — Whisper transcription → word-list matching

**Configuration** (`config.py`):
- `SoundShieldConfig` dataclass with sub-configs for each detector, web, database, pipeline
- All thresholds configurable via environment variables (see `.env.example`)
- Config singleton: `from config import config`
- Runtime threshold tuning via admin dashboard API

**Additional modules:**
- `SpeakerDiarizer` (`speaker_diarizer.py`) — pyannote.audio or pitch-based fallback; classifies adult (<200Hz F0) vs child (>200Hz)
- `Database` (`database.py`) — SQLite persistence (analyses + incidents + audit_log tables); powers history sidebar, comparison, data retention cleanup
- `ReportGenerator` (`report_generator.py`) — JSON/HTML/CSV reports with **Chart.js** visualizations, timeline bar, confidence-weighted severity scoring
- `AuditLogger` (`audit_logger.py`) — append-only audit trail for compliance (uploads, analyses, views, deletions)
- `NotificationManager` (`notifications.py`) — real-time alert system with webhook support, in-app notifications, severity classification
- `Auth` (`auth.py`) — JWT authentication (PyJWT + bcrypt), `UserStore` with SQLite-backed users table, `@require_role` decorator with 3-tier RBAC (viewer/analyst/admin), token generation/validation. Auth disabled by default (`AUTH_ENABLED=False`)
- `Validators` (`validators.py`) — SSRF-protected webhook URL validation (blocks private IPs, file://, localhost), language parameter validation, threshold bounds checking (NaN/Inf/range), audio file magic byte verification
- `StructuredLogging` (`structured_logging.py`) — JSON logging via python-json-logger, `ContextVar`-based correlation IDs for request tracing, `StepTimer` context manager for pipeline step timing, configurable log format (json/text)
- `ModelOptimizer` (`model_optimizer.py`) — ONNX export for HuBERT, INT8 quantization, optimized inference, benchmarking
- `LiveAudioProcessor` (`live_monitor.py`) — WebSocket streaming audio analysis with Socket.IO
- `APIError` (`api_errors.py`) — standardized error response format for all API endpoints

ML models are the **default primary path** (`use_advanced=True`). Loading priority chains:
- Whisper: faster-whisper (CTranslate2, INT8) → openai-whisper → disabled
- HuBERT: ONNX Runtime → PyTorch pipeline → disabled
- Heuristics serve as fallback only when models aren't available. Each detection is tagged `ml_backed=True/False`.

**Web Frontend** (`web_app.py` + `templates/index.html` + `static/`):
- Modern dashboard: Tailwind CSS, Alpine.js, Chart.js, wavesurfer.js (all CDN, no build step)
- **SSE progress** via `/progress-stream/<filename>` (replaces polling)
- **WebSocket live monitoring** via Socket.IO `/ws` namespace (optional, requires flask-socketio)
- Waveform with colored incident overlays (violence=red, emotion=orange, cry=blue, neglect=gray, language=purple)
- Severity doughnut, incident type bar, timeline density, emotion radar charts
- Dark mode, RTL support (Hebrew), responsive mobile-first
- Modal reports (replaces `window.open` + `document.write`)
- Async analysis via `/upload-async` + `ThreadPoolExecutor`
- History sidebar, comparison endpoint (`/compare?ids=1,2,3`)
- **Admin dashboard** (`/admin`, `templates/admin.html`) — system health, DB stats, threshold tuning sliders, audit log viewer

**Desktop GUI** (`gui_app.py`):
- Tabbed results: Summary | Charts (matplotlib) | Details
- EN/HE bilingual UI

Static files: `static/css/main.css`, `static/js/{app,upload,waveform,charts,modal}.js`

## API Endpoints

### Legacy endpoints
- `GET /` — Main dashboard
- `GET /admin` — Admin dashboard
- `POST /upload` — File upload + analysis
- `POST /upload-async` — Async analysis
- `GET /progress-stream/<filename>` — SSE progress
- `GET /health` — Health check (also at `/api/v1/health`)
- `GET /history`, `/reports`, `/compare` — Report endpoints

### API v1
- `GET /api/v1/health` — Production health check (models, GPU, disk, version)
- `GET /api/v1/analyses` — List analyses (paginated, filterable)
- `GET /api/v1/analyses/<id>` — Single analysis with incidents
- `DELETE /api/v1/analyses/<id>` — Delete analysis (audit-logged)
- `GET /api/v1/stats` — Database statistics
- `GET /api/v1/audit-log` — Audit log entries
- `POST /api/v1/cleanup` — Data retention cleanup
- `GET /api/v1/config/thresholds` — Read all detector thresholds
- `PUT /api/v1/config/thresholds` — Update a threshold (audit-logged)
- `POST /api/v1/config/thresholds/reset` — Reset to defaults
- `GET /api/v1/system/info` — System information
- `GET /api/v1/notifications` — List notifications (filterable)
- `POST /api/v1/notifications/<id>/read` — Mark as read
- `POST /api/v1/notifications/read-all` — Mark all as read
- `GET /api/v1/webhooks` — List webhooks
- `POST /api/v1/webhooks` — Add webhook
- `DELETE /api/v1/webhooks` — Remove webhook
- `POST /api/v1/auth/login` — Authenticate, receive JWT token
- `POST /api/v1/auth/register` — Create user (admin only)
- `GET /api/v1/auth/users` — List users (admin only)
- `POST /api/v1/batch-upload` — Multi-file batch upload
- `GET /api/v1/batch/<id>/status` — Batch job progress
- `GET /api/v1/analyses/<id>/export?format=pdf` — PDF report export
- `GET /api/v1/logs` — Query structured log entries (filterable)

## Key Technical Details

- **Python 3.10+**, heavy deps: PyTorch, TensorFlow, Transformers, librosa, Whisper, msgpack>=1.0.0
- **FFmpeg required** for Whisper transcription (system dependency, not pip)
- Audio formats: WAV, MP3, M4A, FLAC, AAC, OGG; max 500MB; duration 1s–3600s
- All audio resampled to 16kHz via librosa
- Languages: `en` and `he` — passed to Whisper and word-list matching
- Windows console encoding fix applied in both `main.py` and `gui_app.py`
- Tests use `unittest` (not pytest fixtures); test files generate synthetic audio with numpy+soundfile
- No linter or formatter configured in the repo
- **Docker support**: Multi-stage Dockerfile + docker-compose.yml with named volumes
- **CI/CD**: GitHub Actions workflow with test, lint, security scan, Docker build jobs
- **CORS**: configurable via `CORS_ORIGINS` env var (defaults to `*` for dev)
- All configuration via `config.py` dataclasses + env var overrides (see `.env.example`)
- **Authentication**: JWT-based with bcrypt password hashing; 3 roles (viewer, analyst, admin); disabled by default for dev
- **Security headers**: CSP, X-Frame-Options, HSTS, X-Content-Type-Options via `@app.after_request`
- **Rate limiting**: Flask-Limiter with configurable rates per endpoint category (upload: 10/min, API: 120/min)
- **Input validation**: SSRF protection on webhooks, magic byte verification on uploads, threshold bounds checking
- **Structured logging**: JSON format with correlation IDs for pipeline tracing; `LOG_FORMAT=json|text`
- **PDF export**: fpdf2-based report generation with severity banners, incident tables, detection summaries
- **New deps (Sprint 11-12)**: flask-limiter, fpdf2, PyJWT, bcrypt, python-json-logger

## Test Files

| File | Module | Tests |
|------|--------|-------|
| `tests/test_main.py` | KindergartenRecordingAnalyzer | 11 |
| `tests/test_emotion_detector.py` | EmotionDetector | 14 |
| `tests/test_cry_detector.py` | CryDetector | 10 |
| `tests/test_violence_detector.py` | ViolenceDetector | 13 |
| `tests/test_neglect_detector.py` | NeglectDetector | 12 |
| `tests/test_inappropriate_language_detector.py` | InappropriateLanguageDetector | 7 |
| `tests/test_database.py` | Database | 5 |
| `tests/test_config.py` | Config system | 15 |
| `tests/test_integration.py` | Full pipeline + notifications | 12 |
| `tests/test_api.py` | API endpoints (Flask client) | 39 |
| `tests/test_metrics.py` | MetricsCollector | 14 |
| `tests/test_digest.py` | DigestGenerator | 4 |
| `tests/test_resilience.py` | Retry, CircuitBreaker, MemoryGuard | 15 |
| `tests/test_e2e_web.py` | E2E authenticated web flow | 37 |
| `tests/test_security.py` | Security headers, JWT, RBAC, XSS, SQLi | ~20 |

## Commit Convention

```
<type>(<scope>): <subject>
```
Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `style`, `chore`
