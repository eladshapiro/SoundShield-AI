"""Test-session setup: keep uploads, reports, the SQLite database and the log of the
test run out of the real folders (they used to show up as junk in the dashboard's
history sidebar).  Runs before any test module imports ``config``."""
import atexit
import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix='soundshield-tests-')
os.environ['UPLOAD_FOLDER'] = os.path.join(_TMP, 'uploads')
os.environ['REPORTS_FOLDER'] = os.path.join(_TMP, 'reports')
os.environ['DATABASE_PATH'] = os.path.join(_TMP, 'soundshield-test.db')
os.environ['LOG_FILE'] = os.path.join(_TMP, 'soundshield-test.log')
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
