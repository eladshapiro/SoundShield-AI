Install and set up the SoundShield-AI project for development.

Steps:
1. Check Python version (needs 3.10+)
2. Install requirements with `pip install --user -r requirements.txt`
   - IMPORTANT: ensure msgpack>=1.0.0 is installed (required for librosa compatibility)
3. Create required directories: uploads/, reports/, models/, templates/
4. Check if FFmpeg is installed (required for Whisper transcription) — if not, warn the user
5. Copy `.env.example` to `.env` if it doesn't exist
6. Run `python3 install.py` to validate the installation
7. Run `python3 -m pytest tests/ -v` to verify tests pass (expect 110 tests)
8. Report what worked and what failed
