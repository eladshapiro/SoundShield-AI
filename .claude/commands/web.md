Start the SoundShield-AI web interface.

Run: `python3 web_app.py`

This starts a Flask server with:
- Main dashboard at http://localhost:5000
- Admin dashboard at http://localhost:5000/admin (threshold tuning, system health, audit log)
- API v1 endpoints at http://localhost:5000/api/v1/
- WebSocket live monitoring on /ws namespace (if flask-socketio installed)
- File upload, real-time SSE progress tracking, results display, report download
- Health check at http://localhost:5000/health

Configuration via environment variables (see .env.example) or config.py:
- WEB_PORT (default 5000), WEB_HOST (default 0.0.0.0), WEB_DEBUG, CORS_ORIGINS

After starting, inform the user the server is running and how to access it.
If port 5000 is already in use, check what's using it with `lsof -i :5000` and inform the user.
