Debug a specific detector's behavior on an audio file.

Usage: /debug-detector <detector_name> [audio_file]

Arguments from $ARGUMENTS: detector name (emotion, cry, violence, neglect, language, advanced) and optional audio file path.

Steps:
1. Parse detector name and audio file from $ARGUMENTS
2. If no audio file given, generate a synthetic test audio using numpy + soundfile (like tests/test_integration.py does)
3. Write a small Python script that:
   - Loads the audio with `AudioAnalyzer().load_audio(path)` or `librosa.load(path, sr=16000)`
   - Instantiates the target detector class
   - Runs detection and prints full results with all features/scores
   - For emotion: `EmotionDetector` -> `calculate_emotion_features` + `detect_emotion` + `analyze_segment_emotions`
   - For cry: `CryDetector` -> `detect_cry_segments` + `detect_response_to_cry` + `measure_response_time` + `aggregate_cry_episodes`
   - For violence: `ViolenceDetector` -> `detect_violence_segments`
   - For neglect: `NeglectDetector` -> `detect_neglect_patterns`
   - For language: `InappropriateLanguageDetector` -> `detect_inappropriate_language`
   - For advanced: `AdvancedAnalyzer` -> `load_models` + `comprehensive_analysis`
4. Run the script and analyze output
5. Show current config thresholds from `config.py` for this detector
6. Identify threshold values that may need tuning, features that aren't discriminating well, or bugs
7. Clean up the temporary script
8. Suggest running `python3 benchmark.py` to check performance impact
