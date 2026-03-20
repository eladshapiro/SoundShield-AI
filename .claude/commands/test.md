Run the test suite for SoundShield-AI.

Steps:
1. Run the full test suite: `python3 -m pytest tests/ -v`
2. Expected: 110 tests pass (test_main, test_emotion_detector, test_cry_detector, test_violence_detector, test_neglect_detector, test_inappropriate_language_detector, test_database, test_config, test_integration, test_api)
3. If unit tests pass, optionally run the system test: `python3 run_system_test.py`
4. Report results — summarize passes, failures, and any errors
5. If there are failures, read the failing test code and the relevant module code, then suggest fixes

Test file locations:
- Unit tests: `tests/test_*.py`
- Integration tests: `tests/test_integration.py` (full pipeline, config propagation, notifications)
- API smoke tests: `tests/test_api.py` (Flask test client, all API v1 endpoints)
- System test: `run_system_test.py`
- 5-scenario E2E: `run_5_tests.py`
