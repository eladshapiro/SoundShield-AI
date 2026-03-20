Add support for a new language to SoundShield-AI.

Usage: /add-language <language_code>

Arguments from $ARGUMENTS: ISO 639-1 language code (e.g., "ar" for Arabic, "ru" for Russian).

Files that need changes:
1. **config.py**: Add language code to `PipelineConfig.supported_languages` tuple
2. **main.py**: Verify `SUPPORTED_LANGUAGES` uses config (it should via `config.pipeline.supported_languages`)
3. **inappropriate_words.json**: Add a new top-level key with inappropriate word list for the language
4. **inappropriate_language_detector.py**: Ensure Whisper language parameter passes through correctly (it already supports most languages)
5. **report_generator.py**: Add translations dict entries for the new language (severity levels, emotion names, violence types, cry intensity, response quality)
6. **gui_app.py**: Add UI translations in `self.translations` dict in `ModernGUI.__init__`
7. **web_app.py**: Add language option to the web interface and progress message translations
8. **templates/index.html**: Add language option in the Alpine.js translation dict
9. **templates/admin.html**: Add language option in the admin dashboard translation dict

Steps:
1. Read $ARGUMENTS for the language code
2. Verify Whisper supports this language
3. Make all the changes listed above
4. Run tests to verify nothing broke: `python3 -m pytest tests/ -v`
5. Test with `python3 example_usage.py` to verify the pipeline still works
