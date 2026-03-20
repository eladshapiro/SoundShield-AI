Analyze and tune detection thresholds for a specific detector.

Usage: /tune-thresholds <detector_name>

Arguments from $ARGUMENTS: detector name (emotion, cry, violence, neglect).

Steps:
1. Read the detector source file AND config.py to find all threshold definitions
2. List every threshold with its current value (from config.py), env var override name, and what it controls
3. Generate synthetic test audio samples that should trigger and should NOT trigger the detector:
   - True positive: audio designed to trigger detection (high energy, cry-frequency content, etc.)
   - True negative: audio that should not trigger (calm speech, silence, background noise)
4. Run the detector on each sample, capture confidence/severity scores
5. Identify thresholds that are too aggressive (false positives) or too lenient (false negatives)
6. Propose specific threshold adjustments with rationale
7. Ask the user before applying changes
8. Changes can be applied either:
   - In config.py default values (permanent)
   - Via .env file (per-deployment)
   - Via admin dashboard API: `PUT /api/v1/config/thresholds` (runtime, non-persistent)

Threshold locations (all centralized in config.py):
- EmotionDetectorConfig: anger, stress, calm, aggression energy/pitch/spectral thresholds
- CryDetectorConfig: frequency range, energy, voiced ratio, AM depth, response window
- ViolenceDetectorConfig: shouting, aggressive, threatening, physical violence thresholds + energy gate
- NeglectDetectorConfig: unanswered cry duration, silence, interaction, response window

Run `python3 benchmark.py --duration 5` to measure detector performance after changes.
