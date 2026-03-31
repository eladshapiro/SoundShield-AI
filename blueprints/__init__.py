"""
SoundShield-AI Flask Blueprints

Planned decomposition of web_app.py (2592 lines) into modular blueprints:

- api_v1: All /api/v1/* routes (analyses, batch, export, stats, config, system)
- auth: Authentication routes (/api/v1/auth/*, /login page)
- admin: Admin dashboard and user management
- legacy: Legacy web routes (upload, reports, history, compare)

Each blueprint will import shared state (db, analyzer, audit, notifications)
from the main app module.

This package was created in Sprint 16 as preparation for incremental
route extraction in future sprints.
"""
