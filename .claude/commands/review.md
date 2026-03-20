Review recent code changes in the SoundShield-AI repository.

Steps:
1. Run `git diff` to see unstaged changes, and `git diff --cached` for staged changes
2. If no local changes, run `git log --oneline -10` and review the most recent commits with `git diff HEAD~1`

Review criteria specific to this project:
- **Pipeline integrity**: Does the change break the 7-step analysis pipeline in main.py?
- **Config integration**: New thresholds must go in config.py dataclasses, not hardcoded in detector files
- **Optional module pattern**: New imports that might not be available must be wrapped in try/except with availability flags (see ADVANCED_MODELS_AVAILABLE pattern)
- **Detector consistency**: New/modified detectors should follow the sliding-window pattern (segment_length, hop_length, features dict, confidence/severity scores)
- **Bilingual support**: UI text and report elements need both EN and HE translations
- **Result dict structure**: Each detector returns segments with start_time, end_time, duration, confidence, severity
- **Report generation**: New result types must be handled in report_generator.py (JSON, HTML, CSV)
- **Frontend sync**: Changes to analysis results need updates in both gui_app.py and web_app.py
- **API consistency**: New endpoints follow /api/v1/ prefix, use APIError for errors, return {data: ...} format
- **Audit logging**: Sensitive operations (uploads, deletions, config changes) must be audit-logged
- **Test coverage**: New modules need test files in tests/

Report findings organized by severity (critical / warning / suggestion).
