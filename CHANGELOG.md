# Changelog

All notable changes to SoundShield-AI are documented here.

## [3.1.0] - 2026-11-08

### Changed — learned decision heads: far fewer false alarms
The v3.0 anger / aggression / shouting decisions were threshold rules over a few
model outputs. On 20-minute held-out recordings they raised 64 false anger alerts
and 58 false shouting alerts per hour, and on recordings with real kids at play
the shouting rule fired 84-108 times per hour. v3.1 adds two small logistic
regression heads (`event_heads.py`, weights in `heads/*.json`) on the embeddings
the models already compute for every 5 s window, so there is no extra model and
almost no extra compute:

- **`anger` head** (wav2vec2 MSP-Dim + HuBERT embeddings, dims, HuBERT probs):
  drives concerning-emotion `anger` and, with the arousal floor, the violence
  detector's `aggressive_tone`. Held-out clips: F1 0.65 → 0.85, false-positive
  rate 13.7% → 2.6% (acted happy speech 69% → 1%, surprised 83% → 0%, fearful
  42% → 1%, read speech 9% → 0%, Hebrew read speech 0% → 0%).
- **`scream` head** (+ AST embedding and tag groups): drives `shouting`; tells
  fear / pain / anger screams apart from joyful screams, kids at play and cries.
  Kids-at-play clips flagged 11% → 0%, infant cries 26% → 0.4%.
- On the 8 held-out 20-minute montages (the v3.0 report's seeds 0/1 plus
  "kindergarten" versions), through the production detector code:
  anger recall 0.78 → 0.88 at 64 → 13 false alarms/hour; shouting recall
  0.68 → 0.70 at 58 → 4.9 false alarms/hour; aggressive tone recall 0.50 → 0.88
  at 42 → 12 false alarms/hour (`evaluation/results/heads.md`).
- Heads are trained and their thresholds chosen on the dev split only
  (`python -m evaluation.heads`: two speaker folds with their own montages,
  cross-fitted). Each head falls back to the v3.0 rule when its file is missing,
  a model it needs is not loaded, or `USE_LEARNED_HEADS=false`. Thresholds can be
  moved with `EMOTION_ANGER_HEAD_THRESHOLD` / `VIOLENCE_SCREAM_HEAD_THRESHOLD`
  (dev trade-off curves in `.env.example` and `evaluation/results/heads_train_*.json`).
- `AudioTagger.tag()` / `EmotionAnalyzer.timeline()` can keep the pooled
  embeddings of every window (`embeddings=True`, thread-local forward hooks; the
  probabilities they return are unchanged).

### Added
- Kids-at-play test data: 637 UrbanSound8K `children_playing` clips
  (`python -m evaluation.fetch_urbansound_children`, dataset `us8k_children`) and
  "kindergarten" montages (`python -m evaluation.montage --kids-share 0.5`).
- `evaluation/heads.py` (extract / train / report), `tests/test_event_heads.py`,
  `tests/test_oom_resilience.py` (293 tests in total).

### Fixed
- A GPU without enough free memory for one batch (a smaller card, or one shared
  with another program) made the emotion models fail and the whole analysis
  quietly fell back to the v2 heuristics. The tagger and both emotion models now
  retry an out-of-memory batch in halves (`cuda_compat.halve_on_oom`; identical
  results), so they keep running on a busy GPU.
- The test suite wrote uploads, reports, database rows and log lines into the
  real `uploads/`, `reports/` and `soundshield.db` (the "e2e_test.wav" entries in
  the dashboard history); `tests/conftest.py` now points them at a temp dir.
- `ReportGenerator()` ignored `REPORTS_FOLDER` and always wrote to `reports/`.

### Known limits
- Anger is still learned from acted English speech (RAVDESS, CREMA-D); acted
  disgust is the largest remaining false-alarm source (CREMA-D 13%).
- On very short isolated scream clips the scream head trades recall for precision
  (0.73 → 0.46 recall, 6% → 1.3% false positives); on whole recordings its recall
  matches v3.0. Lower `VIOLENCE_SCREAM_HEAD_THRESHOLD` (e.g. 0.6) for more recall.
- `potential_physical_violence` (impact sounds: doors, toys, knocks) is unchanged
  at ~7 false alarms/hour on the montages, and `aggressive_tone` repeats most
  `anger` alerts as a second incident (12 false/hour, the same windows).
- The new kindergarten montages show the (unchanged) cry detector raising 21-33
  false cries/hour around kids at play; a learned cry head is the next step.
- End-to-end check through `main.py` (`montage_20min_{,kg_}seed0_v31.json`) agrees
  with the report within one false alarm per montage.

### Changed — dashboard redesign (2026-09-23, first released in 3.1.0)
- Web dashboard rebuilt as a single front end (`templates/index.html`,
  `static/css/main.css`, `static/js/app.js`): Apple-style design system with its
  own tokens (no Tailwind), Alpine stores, a wavesurfer recording strip with
  severity-coloured incident markers, findings list, Whisper transcript, neural
  voice-signal meters, dark mode, Hebrew RTL and a phone layout. The
  `upload/waveform/charts/modal.js` modules that collided with the inline
  template code (duplicate `const translations` → SyntaxError, dead Analyze
  button) are removed; the Chart.js charts are replaced by the verdict meter and
  signal bars.
- Upload flow uses `/upload-async` with SSE progress and a `/job-status`
  backstop, client-side validation and a working Cancel; saved reports reopen
  from the sidebar into the same view.

### Fixed
- `/upload` and `/job-status` return the same full payload
  (`_build_analysis_payload`): incidents with numeric times, audio clips,
  metadata and recommendations.
- `/reports` skips truncated report files instead of failing with 500.
- `/progress-stream` ends immediately for jobs that already finished.
- Report JSON is written atomically with a numpy-safe encoder.
- Summary `total_incidents` counts unanswered cries and ignored distress from
  `neglect_analysis`; `critical_incidents` includes critical inappropriate words.
- `analyze_with_whisper` was gated on a model attribute that faster-whisper only
  sets lazily, so the report's transcript section was empty on the first analysis.

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
