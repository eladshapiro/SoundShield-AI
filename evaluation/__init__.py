"""Real-audio evaluation suite for SoundShield-AI detectors.

Public datasets are downloaded to ``data/raw`` (git-ignored), unpacked to
``data/<name>`` and turned into per-task manifests.  ``run_eval`` runs the
production detector code paths on every clip and writes precision / recall /
F1 tables to ``evaluation/results``.
"""
