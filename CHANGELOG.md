# Changelog

All notable changes to SoundShield-AI are documented here.

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
