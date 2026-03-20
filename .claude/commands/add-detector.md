Guide and implement adding a new detector module to SoundShield-AI.

The user wants to add a new detector. Ask them what kind of detection they need if not specified in $ARGUMENTS.

Follow the existing detector pattern used by EmotionDetector, CryDetector, ViolenceDetector, NeglectDetector:

1. **Add config** (`config.py`):
   - Create a new dataclass (e.g., `NewDetectorConfig`) with all thresholds as env-overridable fields
   - Add it to `SoundShieldConfig` as a field
   - Update `.env.example` with the new env var names

2. **Create the detector module** (e.g., `new_detector.py`):
   - Import and use config: `from config import config`
   - Use `logging` module (not print())
   - Class with `__init__` that reads thresholds from config
   - Main detection method that takes `(audio: np.ndarray, sr: int)` and returns `List[Dict]` of segments
   - Each segment dict has: start_time, end_time, duration, confidence, severity, features, details
   - Private `_calculate_*_features(segment, sr)` method extracting librosa features
   - Private `_is_*_segment(features)` method applying thresholds
   - Confidence calculation method
   - Use sliding window analysis: configurable segment_length and hop_length in seconds

3. **Wire into the orchestrator** (`main.py`):
   - Add import in the try/except block at top (optional module pattern if appropriate)
   - Add availability flag if optional
   - Initialize in `KindergartenRecordingAnalyzer.__init__`
   - Add analysis step in `analyze_audio_file` (increment total_steps)
   - Add results to the `analysis_results` dict

4. **Wire into report generator** (`report_generator.py`):
   - Add translations for Hebrew in `self.translations`
   - Handle new results in `generate_comprehensive_report`

5. **Add to frontends** (both `gui_app.py` and `web_app.py`):
   - Display results in results sections

6. **Write tests**:
   - Unit test in `tests/test_<detector>.py` following the pattern in existing test files
   - Add a scenario to `run_5_tests.py` if appropriate

7. **Add notification support** (`notifications.py`):
   - Add incident type label in `notify_critical_incident.type_labels`

After creating, run `python3 -m pytest tests/ -v` to verify nothing broke (expect 110+ tests).
