# Changelog

All notable changes to SoundShield-AI are documented here.

## [3.0.0] - 2026-09-09

### Changed — ML-first detection, validated on real audio
The v2.x detectors were tuned on synthetic tones. Measured on ~7,000 public,
labelled real-world clips (`evaluation/`), the spectral heuristics flagged 76% of
non-cry sounds as crying, missed 87% of screams, and could never detect a staff
response (the response test compared a spectral centroid against 100 Hz). v3.0
replaces them as the primary path with pretrained audio models and keeps them
only as a fallback when the models are unavailable.

- **AudioSet event tagger** (`audio_tagger.py`, Audio Spectrogram Transformer,
  527 classes): window-level timeline of cry / scream / speech / laughter / impact
  probabilities, computed once per recording on the GPU when available.
  Cry detection on held-out clips: precision 0.97, recall 0.88 (v2.5: 0.17 / 0.91).
- **Dimensional emotion model** (`emotion_models.py`, audeering wav2vec2 MSP-Dim):
  arousal / dominance / valence per 5 s window — language-independent, so it works
  for Hebrew speech where the English HuBERT classifier does not. Anger rule =
  high arousal + high dominance + (low valence or HuBERT agrees).
- **Violence detector**: `shouting` from the scream/shout tags, `aggressive_tone`
  from speech-gated arousal/dominance/valence, `potential_physical_violence` from
  impact tags. Held-out F1 0.77 (v2.5: 0.08).
- **Staff-response / neglect logic** now uses the tagger's speech probability with
  a "speech must dominate crying" rule (held-out F1 0.98 for adult-speech
  presence; v2.5: 0.00).
- **Per-language Whisper**: Hebrew uses `ivrit-ai/whisper-large-v3-turbo-ct2`
  (21.4% WER on FLEURS he_il vs 66.1% for `base`); English uses `large-v3-turbo`
  (4.4% vs 5.8% WER on LibriSpeech). faster-whisper runs on the GPU (float16)
  with CUDA-12 runtime pre-loading (`cuda_compat.py`). Transcription runs once
  and is shared with the inappropriate-language detector (previously Whisper
  `base` was loaded and run a second time).
- Web app and desktop GUI now delegate to `KindergartenRecordingAnalyzer`
  instead of carrying their own copies of the pipeline.
- All operating points live in `config.py` / `.env` and were chosen on the dev
  split of the evaluation suite; reported numbers are from the held-out split.
- Severity of `aggression` (dimensional model only) is one level below `anger`
  (both models agree). `EMOTION_MIN_WINDOWS` can require sustained anger.

### Added
- `evaluation/` — dataset preparation, per-task manifests, runners, metrics,
  dev-tuned threshold sweep (`fusion.py`), long-recording montage test
  (`montage.py`), results in `evaluation/results/`.
- `tests/test_audio_tagger.py`, `tests/test_emotion_models.py`,
  `tests/test_ml_integration.py` (32 tests, no model download needed).
- `requirements-eval.txt`.

### Fixed
- HuBERT pipeline ran on CPU even with a GPU present.
- `analyze_with_whisper` in the language detector loaded its own Whisper model
  on every call.

## [2.5.0] - 2026-03-31

### Added — Sprints 17-20
- Prometheus-compatible metrics endpoint (`GET /metrics`) with counters, histograms, error rates
- Pipeline step timing percentiles (`GET /api/v1/metrics/pipeline-timing`)
- Error rate tracking with sliding window (`GET /api/v1/metrics/error-rates`)
- CSV batch export with date/risk filters (`GET /api/v1/export/csv`)
- Analysis comparison API for up to 5 analyses (`GET /api/v1/analyses/compare`)
- Daily and weekly digest generation (`GET /api/v1/digest/daily`, `/weekly`)
- Retry decorator with exponential backoff (`resilience.py`)
- Circuit breaker pattern for ML model resilience (`resilience.py`)
- Memory guard with psutil monitoring (`GET /api/v1/metrics/memory`)
- CHANGELOG.md and VERSION file
- `GET /api/v1/version` endpoint with build info
- Security test suite (`tests/test_security.py`)

## [2.0.0] - 2026-03-31

### Added — Sprints 13-16
- JWT authentication with bcrypt password hashing (`auth.py`)
- Role-based access control: viewer, analyst, admin (`@require_role`)
- Login page with EN/HE bilingual UI (`templates/login.html`)
- Auth-aware navbar with role badges and logout
- Token refresh (`POST /api/v1/auth/refresh`) and profile (`GET /api/v1/auth/me`)
- Admin user management UI in dashboard (create, role change, deactivate)
- Input validation module (`validators.py`) — SSRF protection, threshold bounds, magic bytes
- Structured JSON logging with correlation IDs (`structured_logging.py`)
- Request ID middleware (`X-Request-ID` header on all responses)
- API pagination for audit log and notifications
- Detection accuracy tuning calibrated on real-world audio samples
- "Distress" emotion category for child crying detection
- Blueprint directory structure prep
- E2E authenticated web flow test suite (37 tests)

## [1.5.0] - 2026-03-31

### Added — Sprints 11-12
- Security headers (CSP, X-Frame-Options, HSTS, X-Content-Type-Options)
- Flask-Limiter rate limiting (configurable per endpoint)
- SECRET_KEY configuration from environment
- Batch file upload (`POST /api/v1/batch-upload`) with job tracking
- PDF report export using fpdf2 (`GET /api/v1/analyses/<id>/export`)
- Comprehensive `.env.example` with 110+ documented variables

## [1.1.0] - 2026-03-30

### Added — Sprints 1-10
- 7-detector audio analysis pipeline (emotion, cry, violence, neglect, language, advanced ML, speaker diarization)
- ML-first design: HuBERT emotion + Whisper transcription (with heuristic fallback)
- Modern web dashboard with Tailwind CSS, Alpine.js, Chart.js, wavesurfer.js
- Real-time SSE progress streaming
- WebSocket live audio monitoring
- Admin dashboard with threshold tuning
- SQLite persistence (analyses, incidents, audit log)
- Notification system with webhook support
- ONNX model optimization and benchmarking
- Docker + docker-compose support
- GitHub Actions CI/CD pipeline
- Centralized configuration via dataclasses + env vars
- API v1 with 30+ endpoints
- 110 unit and integration tests
