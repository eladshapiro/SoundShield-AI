Analyze an audio file using the SoundShield-AI pipeline.

Usage: /analyze <path_to_audio_file> [language]

Arguments:
- $ARGUMENTS contains the audio file path and optional language (en/he, default: en)

Steps:
1. Parse the audio file path and language from $ARGUMENTS
2. Verify the file exists and has a supported format (.wav, .mp3, .m4a, .flac, .aac, .ogg)
3. Run: `python3 main.py <audio_file>`
4. Check if reports were generated in the reports/ directory
5. Read the latest JSON report and summarize findings: emotions, cry events, violence, neglect, inappropriate language
6. Report severity level and any critical incidents
7. If critical incidents found, note that the notification system would alert via webhooks in production
