"""
Web Application for Kindergarten Recording Analyzer
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, Response
from flask_cors import CORS
import os
import sys
import json
import time
import shutil
import platform
import queue
from dataclasses import asdict, fields
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
from main import KindergartenRecordingAnalyzer
from config import (config, AudioAnalyzerConfig, CryDetectorConfig,
                     ViolenceDetectorConfig, EmotionDetectorConfig,
                     NeglectDetectorConfig)
from api_errors import register_error_handlers, APIError
from audit_logger import AuditLogger
from notifications import notifications
from validators import validate_webhook_url, validate_language, validate_threshold_value, validate_audio_file
from auth import UserStore, generate_token, require_role, get_current_user

# Structured logging with correlation IDs
try:
    from structured_logging import setup_logging, set_correlation_id, get_correlation_id, StepTimer
    _structured_logging_available = True
except ImportError:
    _structured_logging_available = False

# Import database
try:
    from database import Database
    db = Database()
except ImportError:
    db = None

# Audit logger
audit = AuditLogger()

# User authentication
user_store = UserStore()

# Initialize structured logging (conditional)
if _structured_logging_available:
    setup_logging()

app = Flask(__name__, static_folder='static', static_url_path='/static')

# CORS: restrict origins in production via CORS_ORIGINS env var
_cors_origins = config.web.cors_origins
if _cors_origins == '*':
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": [o.strip() for o in _cors_origins.split(',')]}})

register_error_handlers(app)

# Security: SECRET_KEY
app.config['SECRET_KEY'] = config.security.secret_key

# Rate limiting (graceful: disabled if flask-limiter not installed)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[config.security.rate_limit_default],
        storage_uri="memory://",
    )
except ImportError:
    limiter = None


def rate_limit(limit_string):
    """Apply rate limit if flask-limiter is available."""
    def decorator(f):
        if limiter is not None:
            return limiter.limit(limit_string)(f)
        return f
    return decorator


# Security headers
@app.after_request
def add_security_headers(response):
    if config.security.enable_security_headers:
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.tailwindcss.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "media-src 'self' blob:"
        )
        if not app.debug:
            response.headers['Strict-Transport-Security'] = f'max-age={config.security.hsts_max_age}; includeSubDomains'
    return response

# Configuration — driven by config.py
UPLOAD_FOLDER = config.web.upload_folder
REPORTS_FOLDER = config.web.reports_folder
ALLOWED_EXTENSIONS = set(config.web.allowed_extensions)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.web.max_content_length

# Thread pool for async analysis
executor = ThreadPoolExecutor(max_workers=config.web.max_workers)

# Initialize analyzer
analyzer = None

# Progress tracking - use dictionary to support multiple concurrent analyses
# Key: filename, Value: progress dict
progress_tracking = {}
# SSE event queues per filename
sse_queues = {}
# Async job results
job_results = {}
# Batch job tracking
batch_jobs = {}
# Default progress for backward compatibility
current_progress = {
    'status': 'idle',
    'message': '',
    'step': 0,
    'total_steps': 7
}

def safe_print(msg):
    """Print to both stdout and stderr to ensure visibility"""
    print(msg, flush=True)
    import sys
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def update_progress(status, message, step=None, filename=None):
    """Update progress status for a specific analysis

    Args:
        status: Progress status
        message: Progress message
        step: Current step number
        filename: Optional filename to track progress per analysis
    """
    global current_progress, progress_tracking

    if filename:
        # Use per-analysis progress tracking
        if filename not in progress_tracking:
            progress_tracking[filename] = {
                'status': 'analyzing',
                'message': '',
                'step': 0,
                'total_steps': 7
            }

        progress_tracking[filename]['status'] = status
        progress_tracking[filename]['message'] = message
        if step is not None:
            progress_tracking[filename]['step'] = step

        # Push SSE event
        if filename in sse_queues:
            event_data = {
                'status': status,
                'message': message,
                'step': step or progress_tracking[filename].get('step', 0),
                'total_steps': 7
            }
            for q in sse_queues[filename]:
                try:
                    q.put_nowait(event_data)
                except queue.Full:
                    pass

        # Also update global for backward compatibility (latest analysis)
        current_progress['status'] = status
        current_progress['message'] = message
        if step is not None:
            current_progress['step'] = step
    else:
        # Update global progress only
        current_progress['status'] = status
        current_progress['message'] = message
        if step is not None:
            current_progress['step'] = step

    # Print to both stdout and stderr to ensure visibility
    msg = f"Progress: {status} - {message}"
    safe_print(msg)

def run_analysis_with_progress(analyzer, file_path, language, progress_callback):
    """Run analysis with progress updates"""
    import threading
    
    # Translations for progress messages
    translations = {
        'he': {
            'step1': 'שלב 1: ניתוח אודיו בסיסי',
            'step2': 'שלב 2: זיהוי רגשות',
            'step3': 'שלב 3: זיהוי בכי',
            'step4': 'שלב 4: זיהוי אלימות',
            'step5': 'שלב 5: זיהוי הזנחה',
            'step6': 'שלב 6: ניתוח מתקדם עם מודלי ML',
            'step7': 'שלב 7: זיהוי שפה לא הולמת',
            'generating_report': 'יוצר דוח...'
        },
        'en': {
            'step1': 'Step 1: Basic audio analysis',
            'step2': 'Step 2: Emotion detection',
            'step3': 'Step 3: Baby cry detection',
            'step4': 'Step 4: Violence detection',
            'step5': 'Step 5: Neglect detection',
            'step6': 'Step 6: Advanced analysis with ML models',
            'step7': 'Step 7: Inappropriate language detection',
            'generating_report': 'Generating report...'
        }
    }
    
    lang_dict = translations.get(language, translations['en'])
    
    # Step 1: Basic audio analysis
    progress_callback(1, lang_dict['step1'])
    safe_print(f"🔄 {lang_dict['step1']}")
    audio_analysis = analyzer.audio_analyzer.analyze_audio_file(file_path)
    safe_print(f"✅ {lang_dict['step1']} completed")
    
    # Step 2: Emotion detection
    progress_callback(2, lang_dict['step2'])
    safe_print(f"🔄 {lang_dict['step2']}")
    emotion_results = analyzer.emotion_detector.analyze_segment_emotions(
        audio_analysis['segments'], 
        audio_analysis['sample_rate']
    )
    concerning_emotions = analyzer.emotion_detector.detect_concerning_emotions(emotion_results)
    safe_print(f"✅ {lang_dict['step2']} completed")
    
    # Step 3: Cry detection
    progress_callback(3, lang_dict['step3'])
    safe_print(f"🔄 {lang_dict['step3']}")
    audio, sr = analyzer.audio_analyzer.load_audio(file_path)
    cry_segments = analyzer.cry_detector.detect_cry_segments(audio, sr)
    cry_with_responses = analyzer.cry_detector.detect_response_to_cry(audio, sr, cry_segments)
    safe_print(f"✅ {lang_dict['step3']} completed")
    
    # Step 4: Violence detection
    progress_callback(4, lang_dict['step4'])
    safe_print(f"🔄 {lang_dict['step4']}")
    violence_segments = analyzer.violence_detector.detect_violence_segments(audio, sr)
    safe_print(f"✅ {lang_dict['step4']} completed")
    
    # Step 5: Neglect detection
    progress_callback(5, lang_dict['step5'])
    safe_print(f"🔄 {lang_dict['step5']}")
    neglect_analysis = analyzer.neglect_detector.detect_neglect_patterns(
        audio, sr, cry_segments, violence_segments
    )
    safe_print(f"✅ {lang_dict['step5']} completed")
    
    # Step 6: Advanced analysis
    advanced_analysis = {}
    if analyzer.advanced_analyzer and analyzer.advanced_analyzer.models_loaded:
        progress_callback(6, lang_dict['step6'])
        safe_print(f"🔄 {lang_dict['step6']}")
        try:
            advanced_analysis = analyzer.advanced_analyzer.comprehensive_analysis(file_path, language=language)
            safe_print(f"✅ {lang_dict['step6']} completed")
        except Exception as e:
            safe_print(f"⚠️ Error in advanced analysis: {e}")
    else:
        # Skip step 6 if not available
        safe_print(f"⏭️ Skipping {lang_dict['step6']} (not available)")
    
    # Step 7: Inappropriate language detection
    inappropriate_language = {}
    if analyzer.language_detector:
        progress_callback(7, lang_dict['step7'])
        safe_print(f"🔄 {lang_dict['step7']}")
        try:
            inappropriate_language = analyzer.language_detector.analyze_with_whisper(file_path, language=language)
            
            # Check if there was an error
            if inappropriate_language.get('status') == 'error':
                safe_print(f"⚠️ Error in language detection: {inappropriate_language.get('error', 'Unknown error')}")
            elif inappropriate_language.get('status') == 'whisper_not_installed':
                safe_print(f"⚠️ Whisper not installed: {inappropriate_language.get('error', 'Unknown error')}")
            elif inappropriate_language.get('detected_inappropriate_words', 0) > 0:
                safe_print(f"⚠️ Detected {inappropriate_language['detected_inappropriate_words']} inappropriate words")
            else:
                safe_print("✅ No inappropriate language detected")
            
            safe_print(f"✅ {lang_dict['step7']} completed")
        except Exception as e:
            safe_print(f"⚠️ Error in language detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Skip step 7 if not available
        safe_print(f"⏭️ Skipping {lang_dict['step7']} (not available)")
    
    # Generate report
    progress_callback(7, lang_dict['generating_report'])
    safe_print(f"🔄 {lang_dict['generating_report']}")
    
    # Speaker diarization
    diarization_results = {}
    if hasattr(analyzer, 'speaker_diarizer') and analyzer.speaker_diarizer:
        try:
            speaker_segments = analyzer.speaker_diarizer.get_speaker_segments(file_path)
            diarization_results = analyzer.speaker_diarizer.get_summary(speaker_segments)
            diarization_results['segments'] = speaker_segments
            safe_print(f"Diarization: {diarization_results.get('speaker_count', 0)} speakers")
        except Exception as e:
            safe_print(f"Diarization error: {e}")

    # Track which models were used
    models_used = []
    hubert_used = (analyzer.advanced_analyzer and
                   hasattr(analyzer.advanced_analyzer, 'hubert_loaded') and
                   analyzer.advanced_analyzer.hubert_loaded)
    if hubert_used:
        models_used.append('hubert')
    if analyzer.advanced_analyzer and hasattr(analyzer.advanced_analyzer, 'whisper_loaded') and analyzer.advanced_analyzer.whisper_loaded:
        models_used.append('whisper')
    if hasattr(analyzer, 'speaker_diarizer') and analyzer.speaker_diarizer:
        models_used.append('diarizer')

    # Use HuBERT as primary for emotions when available
    if hubert_used:
        try:
            advanced_emotions = analyzer.advanced_analyzer.detect_concerning_emotions_advanced(file_path)
            if advanced_emotions:
                concerning_emotions = analyzer.emotion_detector.merge_with_advanced_results(
                    concerning_emotions, advanced_emotions
                )
                safe_print(f"HuBERT merged {len(advanced_emotions)} emotion detections")
        except Exception as e:
            safe_print(f"HuBERT emotion merge error: {e}")
    else:
        for e in concerning_emotions:
            e['ml_backed'] = False

    # Compile results
    analysis_results = {
        'file_path': file_path,
        'duration': audio_analysis['duration'],
        'audio_analysis': audio_analysis,
        'emotion_results': emotion_results,
        'concerning_emotions': concerning_emotions,
        'cry_segments': cry_segments,
        'cry_with_responses': cry_with_responses,
        'violence_segments': violence_segments,
        'neglect_analysis': neglect_analysis,
        'advanced_analysis': advanced_analysis,
        'diarization': diarization_results,
        'inappropriate_language': inappropriate_language,
        'models_used': models_used,
        'analysis_timestamp': time.time(),
        'language': language
    }
    
    # Generate report
    report = analyzer.generate_report(analysis_results)
    analysis_results['report'] = report
    
    return analysis_results

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/login')
def login_page():
    """Serve the login page."""
    return render_template('login.html')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/admin')
@rate_limit(config.security.rate_limit_api)
@require_role('admin')
def admin_dashboard():
    """Admin dashboard for system monitoring and threshold tuning."""
    return render_template('admin.html')

@app.route('/upload', methods=['POST'])
@rate_limit(config.security.rate_limit_upload)
@require_role('analyst')
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        language = request.form.get('language', 'en')  # Get language from form

        is_valid, error_msg = validate_language(language)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Initialize analyzer if not already done
            global analyzer
            if analyzer is None:
                update_progress('initializing', 'מאתחל את המערכת... / Initializing system...', 0, filename=filename)
                analyzer = KindergartenRecordingAnalyzer(language=language)
            else:
                # Update language if analyzer already exists
                analyzer.language = language
            
            # Generate correlation ID for this analysis
            if _structured_logging_available:
                import logging as _logging
                cid = set_correlation_id()
                _logging.getLogger('soundshield.api').info(
                    f"Analysis started for {filename}",
                    extra={'event': 'analysis_start', 'filename': filename}
                )

            # Initialize progress tracking for this specific analysis
            # Use filename as unique identifier to avoid race conditions
            global progress_tracking
            progress_tracking[filename] = {
                'status': 'analyzing',
                'message': 'מתחיל ניתוח... / Starting analysis...',
                'step': 0,
                'total_steps': 7
            }
            
            # Also update global for backward compatibility
            current_progress['status'] = 'analyzing'
            current_progress['message'] = 'מתחיל ניתוח... / Starting analysis...'
            current_progress['step'] = 0
            current_progress['total_steps'] = 7
            
            # Create progress callback that updates immediately with filename
            def progress_callback(step, message):
                update_progress('analyzing', message, step, filename=filename)
                # Force flush to ensure update is visible
                import sys
                sys.stdout.flush()
            
            # Run analysis with progress updates
            start_msg = f"Starting analysis of {filename} with language {language}..."
            safe_print(start_msg)
            results = run_analysis_with_progress(analyzer, filepath, language, progress_callback)
            
            # Mark as completed
            update_progress('completed', 'Analysis completed successfully!', 7, filename=filename)

            # Save to database
            if db:
                try:
                    original_name = '_'.join(filename.split('_')[1:])
                    models = ', '.join(results.get('models_used', []))
                    db.save_analysis(
                        filename=filename,
                        original_filename=original_name,
                        analysis_results=results,
                        report=results.get('report', {}),
                        models_used=models
                    )
                except Exception as e:
                    safe_print(f"Database save error: {e}")

            # Clean up progress tracking after a delay
            import threading
            def cleanup_progress():
                import time
                time.sleep(300)  # 5 minutes
                if filename in progress_tracking:
                    del progress_tracking[filename]
            threading.Thread(target=cleanup_progress, daemon=True).start()

            # Return results including inappropriate language
            inappropriate_lang = results.get('inappropriate_language', {})
            inappropriate_words_list = []
            if inappropriate_lang.get('words_by_severity'):
                for severity, words in inappropriate_lang['words_by_severity'].items():
                    inappropriate_words_list.extend(words)
            
            # Include audio clips from the report
            audio_clips = results.get('report', {}).get('audio_clips', [])

            # Collect all incidents with timing for waveform overlays
            all_incidents = []
            for e in results.get('concerning_emotions', []):
                all_incidents.append({
                    'type': 'emotion',
                    'start_time': e.get('start_time', 0),
                    'end_time': e.get('end_time', 0),
                    'severity': e.get('severity', 'low'),
                    'label': e.get('detected_emotion', ''),
                    'confidence': e.get('confidence', 0),
                    'ml_backed': e.get('ml_backed', False),
                })
            for v in results.get('violence_segments', []):
                all_incidents.append({
                    'type': 'violence',
                    'start_time': v.get('start_time', 0),
                    'end_time': v.get('end_time', 0),
                    'severity': v.get('adjusted_severity', 'low'),
                    'label': ', '.join(v.get('violence_types', [])),
                    'confidence': v.get('confidence', 0),
                })
            for c in results.get('cry_with_responses', []):
                all_incidents.append({
                    'type': 'cry',
                    'start_time': c.get('start_time', 0),
                    'end_time': c.get('end_time', 0),
                    'severity': c.get('intensity', 'medium'),
                    'label': 'response' if c.get('response_detected') else 'no_response',
                })
            neglect_a = results.get('neglect_analysis', {})
            for n in neglect_a.get('unanswered_cries', []):
                all_incidents.append({
                    'type': 'neglect',
                    'start_time': n.get('cry_start_time', 0),
                    'end_time': n.get('cry_end_time', 0),
                    'severity': n.get('neglect_severity', 'medium'),
                    'label': 'unanswered_cry',
                })

            return jsonify({
                'success': True,
                'message': 'Analysis completed successfully',
                'filename': filename,
                'results': {
                    'summary': results['report']['summary'],
                    'statistics': results['report']['statistics'],
                    'detailed_findings': results['report'].get('detailed_findings', {}),
                    'metadata': results['report'].get('metadata', {}),
                },
                'incidents': all_incidents,
                'duration': results.get('duration', 0),
                'models_used': results.get('models_used', []),
                'audio_clips': audio_clips,
                'diarization': results.get('diarization', {}),
                'inappropriate_language': {
                    'detected_inappropriate_words': inappropriate_lang.get('detected_inappropriate_words', 0),
                    'words_by_severity': inappropriate_lang.get('words_by_severity', {}),
                    'inappropriate_words': inappropriate_words_list
                } if inappropriate_lang else None
            })
        
        else:
            return jsonify({'error': 'File type not supported'}), 400
    
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/analyze/<filename>')
def analyze_file(filename):
    """Analyze a specific file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'קובץ לא נמצא'}), 404
        
        # Initialize analyzer if not already done
        global analyzer
        if analyzer is None:
            analyzer = KindergartenRecordingAnalyzer()
        
        # Run analysis
        results = analyzer.run_complete_analysis(filepath)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'שגיאה בניתוח: {str(e)}'}), 500

@app.route('/reports')
def list_reports():
    """List all generated reports"""
    try:
        reports = []
        
        # List JSON reports
        for filename in os.listdir(REPORTS_FOLDER):
            if filename.endswith('.json'):
                filepath = os.path.join(REPORTS_FOLDER, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    reports.append({
                        'filename': filename,
                        'date': report_data['metadata']['analysis_timestamp'],
                        'original_file': report_data['metadata']['file_name'],
                        'summary': report_data['summary']
                    })
        
        # Sort by date (newest first)
        reports.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'success': True,
            'reports': reports
        })
    
    except Exception as e:
        return jsonify({'error': f'שגיאה בטעינת דוחות: {str(e)}'}), 500

@app.route('/report/<filename>')
def get_report(filename):
    """Get specific report data"""
    try:
        # Try JSON first
        json_path = os.path.join(REPORTS_FOLDER, filename)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                return jsonify({
                    'success': True,
                    'report': report_data
                })
        
        return jsonify({'error': 'דוח לא נמצא'}), 404
    
    except Exception as e:
        return jsonify({'error': f'שגיאה בטעינת דוח: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_report(filename):
    """Download report file"""
    try:
        # Try different file extensions
        extensions = ['.json', '.html', '.csv']
        
        for ext in extensions:
            filepath = os.path.join(REPORTS_FOLDER, filename.replace('.json', ext))
            if os.path.exists(filepath):
                return send_file(filepath, as_attachment=True)
        
        return jsonify({'error': 'קובץ לא נמצא'}), 404
    
    except Exception as e:
        return jsonify({'error': f'שגיאה בהורדה: {str(e)}'}), 500

@app.route('/audio_clip/<filename>')
def serve_audio_clip(filename):
    """Serve audio clip files"""
    try:
        clip_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        if os.path.exists(clip_path):
            return send_file(clip_path, as_attachment=False, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Audio clip not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving audio clip: {str(e)}'}), 500

@app.route('/progress')
def get_progress():
    """Get current analysis progress
    
    Supports both:
    - /progress - returns latest/global progress
    - /progress?filename=<filename> - returns progress for specific analysis
    """
    filename = request.args.get('filename')
    
    if filename and filename in progress_tracking:
        # Return progress for specific analysis
        return jsonify(progress_tracking[filename])
    else:
        # Return global/latest progress for backward compatibility
        return jsonify(current_progress)

@app.route('/health')
@app.route('/api/v1/health')
def health_check():
    """Production-grade health check endpoint."""
    # Disk space
    disk = shutil.disk_usage('.')
    disk_free_gb = round(disk.free / (1024 ** 3), 2)

    # Model availability
    models = {
        'whisper': False,
        'hubert': False,
    }
    if analyzer and hasattr(analyzer, 'advanced_analyzer') and analyzer.advanced_analyzer:
        aa = analyzer.advanced_analyzer
        models['whisper'] = getattr(aa, 'whisper_model', None) is not None
        models['hubert'] = getattr(aa, 'emotion_model', None) is not None

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'version': config.version,
        'analyzer_initialized': analyzer is not None,
        'models': models,
        'gpu_available': gpu_available,
        'disk_free_gb': disk_free_gb,
        'active_jobs': len(job_results),
        'platform': platform.system(),
    })


@app.route('/progress-stream/<filename>')
def progress_stream(filename):
    """Server-Sent Events endpoint for real-time progress updates."""
    def event_generator():
        q = queue.Queue(maxsize=100)
        # Register this client's queue
        if filename not in sse_queues:
            sse_queues[filename] = []
        sse_queues[filename].append(q)

        try:
            # Send current state immediately
            if filename in progress_tracking:
                data = json.dumps(progress_tracking[filename])
                yield f"event: progress\ndata: {data}\n\n"

            while True:
                try:
                    event_data = q.get(timeout=30)
                    data = json.dumps(event_data)

                    if event_data.get('status') == 'completed':
                        yield f"event: complete\ndata: {data}\n\n"
                        break
                    elif event_data.get('status') == 'error':
                        yield f"event: error\ndata: {data}\n\n"
                        break
                    else:
                        yield f"event: progress\ndata: {data}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            # Cleanup
            if filename in sse_queues and q in sse_queues[filename]:
                sse_queues[filename].remove(q)
                if not sse_queues[filename]:
                    del sse_queues[filename]

    return Response(
        event_generator(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/uploaded-audio/<filename>')
def serve_uploaded_audio(filename):
    """Serve uploaded audio files for waveform visualization."""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            mimetype = 'audio/wav'
            if filename.endswith('.mp3'):
                mimetype = 'audio/mpeg'
            elif filename.endswith('.ogg'):
                mimetype = 'audio/ogg'
            elif filename.endswith('.flac'):
                mimetype = 'audio/flac'
            elif filename.endswith('.m4a'):
                mimetype = 'audio/mp4'
            return send_file(filepath, mimetype=mimetype)
        return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload-async', methods=['POST'])
@rate_limit(config.security.rate_limit_upload)
@require_role('analyst')
def upload_file_async():
    """Handle file upload and start async analysis. Returns job_id immediately."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        language = request.form.get('language', 'en')

        is_valid, error_msg = validate_language(language)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Initialize progress tracking
            progress_tracking[filename] = {
                'status': 'analyzing',
                'message': 'Starting analysis...',
                'step': 0,
                'total_steps': 7
            }

            # Start async analysis
            future = executor.submit(
                _run_analysis_async, filepath, filename, language
            )
            job_results[filename] = {'future': future, 'status': 'running'}

            return jsonify({
                'success': True,
                'job_id': filename,
                'message': 'Analysis started'
            })
        else:
            return jsonify({'error': 'File type not supported'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _run_analysis_async(filepath, filename, language):
    """Run analysis in background thread."""
    global analyzer
    try:
        if analyzer is None:
            update_progress('initializing', 'Initializing system...', 0, filename=filename)
            analyzer = KindergartenRecordingAnalyzer(language=language)
        else:
            analyzer.language = language

        def progress_callback(step, message):
            update_progress('analyzing', message, step, filename=filename)

        results = run_analysis_with_progress(analyzer, filepath, language, progress_callback)

        # Save to database
        if db:
            try:
                original_name = '_'.join(filename.split('_')[1:])
                models = ', '.join(results.get('models_used', []))
                db.save_analysis(
                    filename=filename,
                    original_filename=original_name,
                    analysis_results=results,
                    report=results.get('report', {}),
                    models_used=models
                )
            except Exception as e:
                safe_print(f"Database save error: {e}")

        update_progress('completed', 'Analysis completed!', 7, filename=filename)
        job_results[filename] = {'status': 'completed', 'results': results}
        return results

    except Exception as e:
        update_progress('error', str(e), filename=filename)
        job_results[filename] = {'status': 'error', 'error': str(e)}
        raise


@app.route('/api/v1/batch-upload', methods=['POST'])
@require_role('analyst')
def batch_upload():
    """Upload multiple files for batch analysis."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided', 'hint': 'Use field name "files" for file upload'}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    language = request.form.get('language', 'en')
    if language not in config.pipeline.supported_languages:
        return jsonify({'error': f'Unsupported language: {language}'}), 400

    batch_id = f"batch_{int(time.time())}_{os.urandom(4).hex()}"
    batch_jobs[batch_id] = {
        'status': 'processing',
        'created_at': datetime.now().isoformat(),
        'language': language,
        'total_files': 0,
        'completed': 0,
        'failed': 0,
        'results': {}
    }

    valid_files = []
    for f in files:
        if f.filename == '':
            continue
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        if ext not in ALLOWED_EXTENSIONS:
            batch_jobs[batch_id]['results'][f.filename] = {
                'status': 'rejected',
                'error': f'Unsupported format: .{ext}'
            }
            continue

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{batch_id}_{filename}")
        f.save(filepath)
        valid_files.append((filename, filepath))

    if not valid_files:
        batch_jobs[batch_id]['status'] = 'completed'
        batch_jobs[batch_id]['total_files'] = 0
        return jsonify({'batch_id': batch_id, 'status': 'completed', 'message': 'No valid files to process'}), 200

    batch_jobs[batch_id]['total_files'] = len(valid_files)

    # Submit each file to the thread pool
    for filename, filepath in valid_files:
        batch_jobs[batch_id]['results'][filename] = {'status': 'queued'}
        executor.submit(_process_batch_file, batch_id, filename, filepath, language)

    audit.log('batch_upload', details={
        'batch_id': batch_id,
        'file_count': len(valid_files),
        'language': language
    })

    return jsonify({
        'batch_id': batch_id,
        'status': 'processing',
        'total_files': len(valid_files),
        'rejected': {k: v for k, v in batch_jobs[batch_id]['results'].items() if v.get('status') == 'rejected'}
    }), 202


def _process_batch_file(batch_id, filename, filepath, language):
    """Process a single file in a batch job."""
    global analyzer
    try:
        batch_jobs[batch_id]['results'][filename] = {'status': 'processing'}

        if analyzer is None:
            analyzer = KindergartenRecordingAnalyzer(language=language)

        results = analyzer.analyze_audio_file(filepath)

        # Store in database if available
        analysis_id = None
        if db:
            analysis_id = db.save_analysis(filename, results)

        batch_jobs[batch_id]['results'][filename] = {
            'status': 'completed',
            'analysis_id': analysis_id,
            'severity': results.get('severity_level', 'unknown'),
            'incidents_count': len(results.get('incidents', []))
        }
        batch_jobs[batch_id]['completed'] += 1

    except Exception as e:
        batch_jobs[batch_id]['results'][filename] = {
            'status': 'failed',
            'error': str(e)
        }
        batch_jobs[batch_id]['failed'] += 1
    finally:
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except OSError:
            pass

        # Check if batch is complete
        completed = batch_jobs[batch_id]['completed']
        failed = batch_jobs[batch_id]['failed']
        total = batch_jobs[batch_id]['total_files']
        if completed + failed >= total:
            batch_jobs[batch_id]['status'] = 'completed'


@app.route('/api/v1/batch/<batch_id>/status', methods=['GET'])
def batch_status(batch_id):
    """Get status of a batch processing job."""
    if batch_id not in batch_jobs:
        return jsonify({'error': 'Batch job not found'}), 404

    job = batch_jobs[batch_id]
    return jsonify({
        'batch_id': batch_id,
        'status': job['status'],
        'created_at': job['created_at'],
        'language': job['language'],
        'total_files': job['total_files'],
        'completed': job['completed'],
        'failed': job['failed'],
        'progress': f"{job['completed'] + job['failed']}/{job['total_files']}",
        'results': job['results']
    })


@app.route('/job-status/<job_id>')
def job_status(job_id):
    """Check status of an async analysis job."""
    if job_id not in job_results:
        return jsonify({'error': 'Job not found'}), 404

    job = job_results[job_id]
    if job['status'] == 'completed':
        results = job.get('results', {})
        inappropriate_lang = results.get('inappropriate_language', {})
        return jsonify({
            'status': 'completed',
            'results': {
                'summary': results.get('report', {}).get('summary', {}),
                'statistics': results.get('report', {}).get('statistics', {}),
                'detailed_findings': results.get('report', {}).get('detailed_findings', {}),
            },
            'filename': job_id,
            'inappropriate_language': {
                'detected_inappropriate_words': inappropriate_lang.get('detected_inappropriate_words', 0),
            } if inappropriate_lang else None
        })
    elif job['status'] == 'error':
        return jsonify({'status': 'error', 'error': job.get('error', 'Unknown error')})
    else:
        return jsonify({'status': 'running', 'progress': progress_tracking.get(job_id, {})})


@app.route('/compare')
def compare_analyses():
    """Compare multiple analyses side by side. Usage: /compare?ids=1,2,3"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500

    ids_param = request.args.get('ids', '')
    try:
        analysis_ids = [int(x) for x in ids_param.split(',') if x.strip()]
    except ValueError:
        return jsonify({'error': 'Invalid IDs format'}), 400

    if not analysis_ids:
        return jsonify({'error': 'No IDs provided'}), 400

    data = db.get_comparison_data(analysis_ids)
    return jsonify({'success': True, 'analyses': data})


@app.route('/history')
def analysis_history():
    """Get analysis history from database."""
    if not db:
        return jsonify({'success': True, 'history': []})

    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    history = db.get_analysis_history(limit=limit, offset=offset)
    return jsonify({'success': True, 'history': history})


@app.route('/analysis/<int:analysis_id>')
def get_analysis_by_id(analysis_id):
    """Get a specific analysis from database."""
    if not db:
        return jsonify({'error': 'Database not available'}), 500

    analysis = db.get_analysis(analysis_id)
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    incidents = db.get_incidents(analysis_id)
    return jsonify({
        'success': True,
        'analysis': analysis,
        'incidents': incidents
    })

# =====================================================================
# Authentication endpoints
# =====================================================================

@app.route('/api/v1/auth/login', methods=['POST'])
def auth_login():
    """Authenticate and receive a JWT token."""
    if not request.is_json:
        return jsonify({'error': 'JSON body required'}), 400

    username = request.json.get('username', '')
    password = request.json.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = user_store.authenticate(username, password)
    if not user:
        audit.log('login_failed', details={'username': username},
                  user_ip=request.remote_addr)
        return jsonify({'error': 'Invalid credentials'}), 401

    token = generate_token(user)
    audit.log('login_success', details={'username': username, 'role': user['role']},
              user_ip=request.remote_addr)

    return jsonify({
        'token': token,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'role': user['role']
        }
    })


@app.route('/api/v1/auth/register', methods=['POST'])
@require_role('admin')
def auth_register():
    """Register a new user (admin only)."""
    if not request.is_json:
        return jsonify({'error': 'JSON body required'}), 400

    username = request.json.get('username', '')
    password = request.json.get('password', '')
    role = request.json.get('role', 'viewer')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    try:
        user = user_store.create_user(username, password, role)
        audit.log('user_created', details={'username': username, 'role': role},
                  user_ip=request.remote_addr)
        return jsonify({'success': True, 'user': user}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/v1/auth/me', methods=['GET'])
@require_role('viewer')
def auth_me():
    """Get current user profile."""
    from flask import g
    # If a real token was provided, prefer the decoded token user over anonymous
    user = g.get('current_user', {})
    token_user = get_current_user()
    if token_user:
        user = token_user
    return jsonify({
        'user': {
            'id': user.get('id') or user.get('user_id', 0),
            'username': user.get('username', 'anonymous'),
            'role': user.get('role', 'viewer')
        }
    })


@app.route('/api/v1/auth/refresh', methods=['POST'])
@require_role('viewer')
def auth_refresh():
    """Refresh JWT token (extends expiry)."""
    from flask import g
    # If a real token was provided, prefer the decoded token user over anonymous
    user = g.get('current_user', {})
    token_user = get_current_user()
    if token_user:
        user = token_user

    user_id = user.get('id') or user.get('user_id')
    if not user_id:
        return jsonify({'error': 'Invalid token'}), 401

    # Look up user to ensure they still exist and are active
    db_user = user_store.authenticate_by_id(user_id)
    if not db_user:
        return jsonify({'error': 'User no longer active'}), 401

    new_token = generate_token(db_user)
    return jsonify({
        'token': new_token,
        'user': {
            'id': db_user['id'],
            'username': db_user['username'],
            'role': db_user['role']
        }
    })


@app.route('/api/v1/auth/users', methods=['GET'])
@require_role('admin')
def auth_list_users():
    """List all users (admin only)."""
    return jsonify({'data': user_store.list_users()})


# =====================================================================
# API v1 Endpoints
# =====================================================================

@app.route('/api/v1/analyses')
@require_role('viewer')
def api_list_analyses():
    """List analyses with pagination and filtering."""
    if not db:
        raise APIError('DB_UNAVAILABLE', 'Database not available.', 503)

    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    risk_level = request.args.get('risk_level')
    query = request.args.get('q')

    if query or risk_level:
        results = db.search_analyses(query=query, risk_level=risk_level)
    else:
        results = db.get_analysis_history(limit=limit, offset=offset)

    return jsonify({'data': results, 'count': len(results)})


@app.route('/api/v1/analyses/<int:analysis_id>')
def api_get_analysis(analysis_id):
    """Get a specific analysis with incidents."""
    if not db:
        raise APIError('DB_UNAVAILABLE', 'Database not available.', 503)

    analysis = db.get_analysis(analysis_id)
    if not analysis:
        from api_errors import analysis_not_found
        raise analysis_not_found(analysis_id)

    incidents = db.get_incidents(analysis_id)
    audit.log_report_view(analysis_id, user_ip=request.remote_addr)

    return jsonify({
        'data': {
            'analysis': dict(analysis),
            'incidents': [dict(i) for i in incidents],
        }
    })


@app.route('/api/v1/analyses/<int:analysis_id>/export', methods=['GET'])
@require_role('viewer')
def export_analysis(analysis_id):
    """Export analysis as PDF."""
    export_format = request.args.get('format', 'pdf')
    if export_format != 'pdf':
        return jsonify({'error': f'Unsupported format: {export_format}', 'supported': ['pdf']}), 400

    if not db:
        return jsonify({'error': 'Database not available'}), 503

    analysis = db.get_analysis(analysis_id)
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    results = json.loads(analysis['results_json']) if isinstance(analysis['results_json'], str) else analysis['results_json']

    from report_generator import ReportGenerator
    generator = ReportGenerator(output_dir=app.config['REPORTS_FOLDER'])
    pdf_filename = f"report_{analysis_id}_{int(time.time())}.pdf"
    pdf_path = generator.generate_pdf_report(results, filename=pdf_filename)

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF generation failed. Install fpdf2: pip install fpdf2'}), 500

    audit.log('export_report', details={'analysis_id': analysis_id, 'format': 'pdf'})

    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')


@app.route('/api/v1/analyses/<int:analysis_id>', methods=['DELETE'])
@require_role('admin')
def api_delete_analysis(analysis_id):
    """Delete an analysis and its associated data."""
    if not db:
        raise APIError('DB_UNAVAILABLE', 'Database not available.', 503)

    deleted = db.delete_analysis(analysis_id)
    if not deleted:
        from api_errors import analysis_not_found
        raise analysis_not_found(analysis_id)

    audit.log_deletion('analysis', str(analysis_id),
                       user_ip=request.remote_addr)
    return jsonify({'success': True, 'message': f'Analysis {analysis_id} deleted.'})


@app.route('/api/v1/stats')
def api_stats():
    """System statistics."""
    if not db:
        raise APIError('DB_UNAVAILABLE', 'Database not available.', 503)

    stats = db.get_stats()
    return jsonify({'data': stats})


@app.route('/api/v1/audit-log')
def api_audit_log():
    """Retrieve audit log entries."""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    logs = audit.get_recent_logs(limit=limit, offset=offset)
    return jsonify({'data': logs, 'count': len(logs)})


@app.route('/api/v1/cleanup', methods=['POST'])
@require_role('admin')
def api_cleanup():
    """Trigger data retention cleanup."""
    days = request.json.get('retention_days') if request.is_json else None
    deleted = db.cleanup_old_data(days)
    audit.log('data_cleanup', details={'deleted_count': deleted,
                                        'retention_days': days or config.database.retention_days})
    return jsonify({'success': True, 'deleted': deleted})


# --- Notification endpoints ---

@app.route('/api/v1/notifications')
def api_notifications():
    """Get notifications with optional filters."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    level = request.args.get('level')
    unread = request.args.get('unread', 'false').lower() == 'true'
    items = notifications.get_notifications(limit=limit, offset=offset,
                                            level=level, unread_only=unread)
    return jsonify({
        'data': items,
        'count': len(items),
        'unread_count': notifications.get_unread_count(),
    })


@app.route('/api/v1/notifications/<notification_id>/read', methods=['POST'])
def api_mark_notification_read(notification_id):
    """Mark a notification as read."""
    ok = notifications.mark_read(notification_id)
    if not ok:
        raise APIError('NOTIFICATION_NOT_FOUND', 'Notification not found.', 404)
    return jsonify({'success': True})


@app.route('/api/v1/notifications/read-all', methods=['POST'])
def api_mark_all_notifications_read():
    """Mark all notifications as read."""
    notifications.mark_all_read()
    return jsonify({'success': True})


@app.route('/api/v1/webhooks', methods=['GET'])
def api_list_webhooks():
    """List registered webhook URLs."""
    return jsonify({'data': notifications._webhooks})


@app.route('/api/v1/webhooks', methods=['POST'])
@require_role('admin')
def api_add_webhook():
    """Register a new webhook URL."""
    if not request.is_json or 'url' not in request.json:
        raise APIError('INVALID_REQUEST', 'JSON body with "url" required.', 400)
    url = request.json['url']
    is_valid, error_msg = validate_webhook_url(url)
    if not is_valid:
        raise APIError('INVALID_URL', error_msg, 400)
    notifications.add_webhook(url)
    audit.log('webhook_added', details={'url': url},
              user_ip=request.remote_addr)
    return jsonify({'success': True, 'webhooks': notifications._webhooks})


@app.route('/api/v1/webhooks', methods=['DELETE'])
@require_role('admin')
def api_remove_webhook():
    """Remove a webhook URL."""
    if not request.is_json or 'url' not in request.json:
        raise APIError('INVALID_REQUEST', 'JSON body with "url" required.', 400)
    url = request.json['url']
    is_valid, error_msg = validate_webhook_url(url)
    if not is_valid:
        raise APIError('INVALID_URL', error_msg, 400)
    notifications.remove_webhook(url)
    audit.log('webhook_removed', details={'url': url},
              user_ip=request.remote_addr)
    return jsonify({'success': True, 'webhooks': notifications._webhooks})


# --- Threshold configuration endpoints ---

# Mapping of detector names to config attributes
_DETECTOR_CONFIG_MAP = {
    'audio': 'audio',
    'cry': 'cry',
    'violence': 'violence',
    'emotion': 'emotion',
    'neglect': 'neglect',
}

# Mapping of detector names to their default factory classes
_DETECTOR_DEFAULT_FACTORY = {
    'audio': AudioAnalyzerConfig,
    'cry': CryDetectorConfig,
    'violence': ViolenceDetectorConfig,
    'emotion': EmotionDetectorConfig,
    'neglect': NeglectDetectorConfig,
}


def _serialize_config(cfg_obj):
    """Convert a dataclass config to a JSON-safe dict.

    Tuples are converted to lists so json.dumps does not choke.
    """
    d = asdict(cfg_obj)
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


@app.route('/api/v1/config/thresholds')
def api_get_thresholds():
    """Return all detector thresholds grouped by detector name."""
    data = {}
    for name, attr in _DETECTOR_CONFIG_MAP.items():
        data[name] = _serialize_config(getattr(config, attr))
    return jsonify({'data': data})


@app.route('/api/v1/config/thresholds', methods=['PUT'])
@require_role('admin')
def api_update_threshold():
    """Update a single detector threshold value.

    Expects JSON body: {"detector": "cry", "key": "energy_threshold", "value": 0.1}
    """
    if not request.is_json:
        raise APIError('INVALID_CONTENT_TYPE',
                        'Request must be JSON with Content-Type application/json.', 400)

    body = request.get_json()

    # --- Validate required fields ---
    detector = body.get('detector')
    key = body.get('key')
    value = body.get('value')

    missing = [f for f in ('detector', 'key', 'value') if body.get(f) is None]
    if missing:
        raise APIError('MISSING_FIELDS',
                        f'Missing required fields: {", ".join(missing)}', 400)

    # --- Validate threshold value bounds ---
    is_valid, error_msg = validate_threshold_value(detector, key, value)
    if not is_valid:
        raise APIError('INVALID_VALUE', error_msg, 400)

    # --- Validate detector name ---
    if detector not in _DETECTOR_CONFIG_MAP:
        raise APIError('INVALID_DETECTOR',
                        f'Unknown detector: {detector}. '
                        f'Valid detectors: {", ".join(sorted(_DETECTOR_CONFIG_MAP))}',
                        400)

    cfg_obj = getattr(config, _DETECTOR_CONFIG_MAP[detector])

    # --- Validate key exists on the dataclass ---
    valid_keys = {f.name for f in fields(cfg_obj)}
    if key not in valid_keys:
        raise APIError('INVALID_KEY',
                        f'Unknown key "{key}" for detector "{detector}". '
                        f'Valid keys: {", ".join(sorted(valid_keys))}',
                        400)

    # --- Validate and coerce value type ---
    current_value = getattr(cfg_obj, key)
    try:
        if isinstance(current_value, float):
            coerced = float(value)
        elif isinstance(current_value, int):
            coerced = int(value)
        elif isinstance(current_value, tuple):
            if not isinstance(value, (list, tuple)) or len(value) != len(current_value):
                raise APIError('INVALID_VALUE',
                                f'Key "{key}" expects a list of {len(current_value)} numbers.',
                                400)
            coerced = tuple(type(current_value[i])(value[i])
                            for i in range(len(current_value)))
        else:
            coerced = value
    except (TypeError, ValueError) as exc:
        raise APIError('INVALID_VALUE',
                        f'Cannot coerce value for "{key}": {exc}', 400)

    old_value = current_value
    setattr(cfg_obj, key, coerced)

    # Audit-log the change
    audit.log('threshold_update',
              resource_type='config',
              resource_id=f'{detector}.{key}',
              user_ip=request.remote_addr,
              details={'detector': detector, 'key': key,
                       'old_value': old_value if not isinstance(old_value, tuple) else list(old_value),
                       'new_value': coerced if not isinstance(coerced, tuple) else list(coerced)})

    return jsonify({
        'data': {
            'detector': detector,
            'config': _serialize_config(cfg_obj),
        },
        'message': f'{detector}.{key} updated from {old_value} to {coerced}',
    })


@app.route('/api/v1/config/thresholds/reset', methods=['POST'])
@require_role('admin')
def api_reset_thresholds():
    """Reset all detector thresholds to their default values."""
    for name, attr in _DETECTOR_CONFIG_MAP.items():
        default_cfg = _DETECTOR_DEFAULT_FACTORY[name]()
        setattr(config, attr, default_cfg)

    audit.log('threshold_reset',
              resource_type='config',
              resource_id='all',
              user_ip=request.remote_addr,
              details={'detectors': sorted(_DETECTOR_CONFIG_MAP.keys())})

    data = {}
    for name, attr in _DETECTOR_CONFIG_MAP.items():
        data[name] = _serialize_config(getattr(config, attr))

    return jsonify({
        'data': data,
        'message': 'All detector thresholds reset to defaults.',
    })


@app.route('/api/v1/system/info')
def api_system_info():
    """Return system information: Python version, platform, package
    availability, disk usage, and database statistics."""

    # --- Python / OS ---
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.machine(),
    }

    # --- ML package availability ---
    packages = {}
    for pkg in ('torch', 'whisper', 'transformers', 'librosa', 'tensorflow'):
        try:
            mod = __import__(pkg)
            packages[pkg] = {'available': True,
                             'version': getattr(mod, '__version__', 'unknown')}
        except ImportError:
            packages[pkg] = {'available': False, 'version': None}
    info['packages'] = packages

    # --- Disk usage ---
    try:
        total, used, free = shutil.disk_usage('/')
        info['disk'] = {
            'total_gb': round(total / (1024 ** 3), 2),
            'used_gb': round(used / (1024 ** 3), 2),
            'free_gb': round(free / (1024 ** 3), 2),
            'usage_percent': round(used / total * 100, 1),
        }
    except OSError:
        info['disk'] = None

    # --- Upload folder size ---
    try:
        upload_size = sum(
            os.path.getsize(os.path.join(UPLOAD_FOLDER, f))
            for f in os.listdir(UPLOAD_FOLDER)
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
        )
        info['uploads'] = {
            'folder': UPLOAD_FOLDER,
            'size_mb': round(upload_size / (1024 ** 2), 2),
            'file_count': len([f for f in os.listdir(UPLOAD_FOLDER)
                               if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]),
        }
    except OSError:
        info['uploads'] = None

    # --- Database stats ---
    if db:
        try:
            info['database'] = db.get_stats()
        except Exception:
            info['database'] = None
    else:
        info['database'] = None

    return jsonify({'data': info})


@app.route('/api/v1/logs', methods=['GET'])
def api_recent_logs():
    """Return recent structured log entries from the log file."""
    import json as json_module
    log_file = config.logging_config.log_file
    if not os.path.exists(log_file):
        return jsonify({'data': [], 'message': 'No log file found'})

    limit = request.args.get('limit', 50, type=int)
    level_filter = request.args.get('level', '').upper()
    correlation_filter = request.args.get('correlation_id', '')

    entries = []
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        for line in reversed(lines[-500:]):
            try:
                entry = json_module.loads(line.strip())
                if level_filter and entry.get('level', '') != level_filter:
                    continue
                if correlation_filter and entry.get('correlation_id', '') != correlation_filter:
                    continue
                entries.append(entry)
                if len(entries) >= limit:
                    break
            except (json_module.JSONDecodeError, ValueError):
                continue
    except IOError:
        pass

    return jsonify({'data': entries, 'total': len(entries)})

def create_html_template():
    """Create basic HTML template for the web interface"""
    html_content = '''
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>מערכת ניתוח הקלטות גן ילדים</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 0;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background-color: #f0f4ff;
        }
        
        .upload-icon {
            font-size: 3em;
            color: #667eea;
            margin-bottom: 20px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            transition: all 0.3s ease;
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .upload-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 20px;
            display: none;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .results-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: none;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .result-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .result-card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .severity-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .severity-low { background-color: #d4edda; color: #155724; }
        .severity-medium { background-color: #fff3cd; color: #856404; }
        .severity-high { background-color: #f8d7da; color: #721c24; }
        .severity-critical { background-color: #721c24; color: white; }
        
        .reports-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .reports-list {
            margin-top: 20px;
        }
        
        .report-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .report-actions {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
        }
        
        .btn-primary { background-color: #667eea; color: white; }
        .btn-secondary { background-color: #6c757d; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        
        .btn:hover { opacity: 0.8; }
        
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        .progress-message {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
            font-size: 1.1em;
            color: #667eea;
            font-weight: bold;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .progress-steps {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .step-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .step-indicator.active {
            background-color: #e8f4f8;
            border-left: 4px solid #667eea;
        }
        
        .step-indicator.completed {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }
        
        .step-number {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            background-color: #ddd;
            color: #666;
        }
        
        .step-indicator.active .step-number {
            background-color: #667eea;
            color: white;
        }
        
        .step-indicator.completed .step-number {
            background-color: #28a745;
            color: white;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .language-selector {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .language-selector label {
            font-weight: bold;
            font-size: 1.1em;
            color: #667eea;
        }
        
        .language-selector select {
            padding: 10px 20px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            background: white;
            color: #333;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .language-selector select:hover {
            border-color: #667eea;
        }
        
        .language-selector select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 2em; }
            .results-grid { grid-template-columns: 1fr; }
            .report-item { flex-direction: column; gap: 10px; }
            .language-selector { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>מערכת ניתוח הקלטות גן ילדים</h1>
            <p>Kindergarten Recording Analyzer</p>
        </div>
        
        <div class="language-selector">
            <label for="languageSelect">🌐 שפת הניתוח / Analysis Language:</label>
            <select id="languageSelect">
                <option value="he">עברית (Hebrew)</option>
                <option value="en">English (אנגלית)</option>
            </select>
        </div>
        
        <div class="upload-section">
            <h2>העלאת קובץ אודיו</h2>
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">🎵</div>
                <h3>גרור קובץ אודיו לכאן או לחץ לבחירה</h3>
                <p>תומך בפורמטים: WAV, MP3, M4A, FLAC, AAC, OGG</p>
                <input type="file" id="fileInput" class="file-input" accept=".wav,.mp3,.m4a,.flac,.aac,.ogg">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    בחר קובץ
                </button>
            </div>
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="progress-message" id="progressMessage">
                מתחיל ניתוח... / Starting analysis...
            </div>
            <div class="progress-steps" id="progressSteps" style="display: none;">
                <div class="step-indicator" id="step1">
                    <div class="step-number">1</div>
                    <span>ניתוח אודיו בסיסי / Basic audio analysis</span>
                </div>
                <div class="step-indicator" id="step2">
                    <div class="step-number">2</div>
                    <span>זיהוי רגשות / Emotion detection</span>
                </div>
                <div class="step-indicator" id="step3">
                    <div class="step-number">3</div>
                    <span>זיהוי בכי / Baby cry detection</span>
                </div>
                <div class="step-indicator" id="step4">
                    <div class="step-number">4</div>
                    <span>זיהוי אלימות / Violence detection</span>
                </div>
                <div class="step-indicator" id="step5">
                    <div class="step-number">5</div>
                    <span>זיהוי הזנחה / Neglect detection</span>
                </div>
                <div class="step-indicator" id="step6">
                    <div class="step-number">6</div>
                    <span>ניתוח מתקדם / Advanced analysis</span>
                </div>
                <div class="step-indicator" id="step7">
                    <div class="step-number">7</div>
                    <span>זיהוי שפה לא הולמת / Inappropriate language detection</span>
                </div>
            </div>
        </div>
        
        <div class="results-section" id="resultsSection">
            <h2>תוצאות הניתוח</h2>
            <div id="resultsContent"></div>
        </div>
        
        <div class="reports-section">
            <h2>דוחות קודמים</h2>
            <button class="btn btn-primary" onclick="loadReports()">רענן רשימה</button>
            <div class="reports-list" id="reportsList">
                <p>טוען דוחות...</p>
            </div>
        </div>
    </div>
    
    <script>
        let currentAnalysis = null;
        
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        let progressCheckInterval = null;
        
        function updateProgressDisplay() {
            fetch('/progress')
                .then(response => response.json())
                .then(data => {
                    const progressMessage = document.getElementById('progressMessage');
                    const progressSteps = document.getElementById('progressSteps');
                    
                    // Always show progress if we're checking
                    if (data.status === 'analyzing' || data.status === 'initializing' || data.status === 'completed') {
                        if (progressMessage) {
                            progressMessage.textContent = data.message || 'מעבד... / Processing...';
                        }
                        if (progressSteps) {
                            progressSteps.style.display = 'block';
                        }
                        
                        // Update step indicators
                        const currentStep = data.step || 0;
                        for (let i = 1; i <= 7; i++) {
                            const stepElement = document.getElementById('step' + i);
                            if (stepElement) {
                                if (i < currentStep) {
                                    stepElement.className = 'step-indicator completed';
                                } else if (i === currentStep && currentStep > 0) {
                                    stepElement.className = 'step-indicator active';
                                } else {
                                    stepElement.className = 'step-indicator';
                                }
                            }
                        }
                        
                        // Update progress bar
                        const progressPercent = Math.min((currentStep / data.total_steps) * 100, 100);
                        if (progressFill) {
                            progressFill.style.width = progressPercent + '%';
                        }
                        
                        // If completed, stop checking after a delay
                        if (data.status === 'completed') {
                            setTimeout(() => {
                                if (progressCheckInterval) {
                                    clearInterval(progressCheckInterval);
                                    progressCheckInterval = null;
                                }
                            }, 2000);
                        }
                    } else if (data.status === 'idle') {
                        // Only stop if we're not in the middle of an upload
                        // Don't stop immediately, wait a bit
                    }
                })
                .catch(error => {
                    console.error('Error fetching progress:', error);
                });
        }
        
        function handleFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Get selected language
            const languageSelect = document.getElementById('languageSelect');
            const selectedLanguage = languageSelect.value;
            formData.append('language', selectedLanguage);
            
            // Show loading
            loading.style.display = 'block';
            progressBar.style.display = 'block';
            resultsSection.style.display = 'none';
            
            // Show progress steps immediately
            const progressSteps = document.getElementById('progressSteps');
            progressSteps.style.display = 'block';
            const progressMessage = document.getElementById('progressMessage');
            progressMessage.textContent = 'מתחיל ניתוח... / Starting analysis...';
            
            // Reset progress steps
            for (let i = 1; i <= 7; i++) {
                const stepElement = document.getElementById('step' + i);
                stepElement.className = 'step-indicator';
            }
            
            // Start checking progress every 300ms (more frequent)
            if (progressCheckInterval) {
                clearInterval(progressCheckInterval);
            }
            progressCheckInterval = setInterval(updateProgressDisplay, 300);
            updateProgressDisplay(); // Call immediately
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (progressCheckInterval) {
                    clearInterval(progressCheckInterval);
                    progressCheckInterval = null;
                }
                progressFill.style.width = '100%';
                
                setTimeout(() => {
                    loading.style.display = 'none';
                    progressBar.style.display = 'none';
                    
                    if (data.success) {
                        showResults(data.results);
                        showAlert('ניתוח הושלם בהצלחה!', 'success');
                        loadReports(); // Refresh reports list
                    } else {
                        showAlert('שגיאה בניתוח: ' + data.error, 'error');
                    }
                }, 1000);
            })
            .catch(error => {
                if (progressCheckInterval) {
                    clearInterval(progressCheckInterval);
                    progressCheckInterval = null;
                }
                loading.style.display = 'none';
                progressBar.style.display = 'none';
                showAlert('שגיאה בהעלאת הקובץ: ' + error.message, 'error');
            });
        }
        
        function showResults(results) {
            const summary = results.summary;
            const stats = results.statistics;
            
            let html = `
                <div class="results-grid">
                    <div class="result-card">
                        <h3>הערכה כללית</h3>
                        <p>${summary.overall_assessment}</p>
                        <span class="severity-badge severity-${summary.risk_level}">
                            רמת סיכון: ${summary.risk_level}
                        </span>
                    </div>
                    <div class="result-card">
                        <h3>סה"כ אירועים</h3>
                        <p>${summary.total_incidents}</p>
                    </div>
                    <div class="result-card">
                        <h3>אירועים קריטיים</h3>
                        <p>${summary.critical_incidents}</p>
                    </div>
                    <div class="result-card">
                        <h3>משך הקלטה</h3>
                        <p>${stats.audio_duration_minutes.toFixed(1)} דקות</p>
                    </div>
                </div>
            `;
            
            if (summary.key_findings && summary.key_findings.length > 0) {
                html += '<h3>ממצאים עיקריים:</h3><ul>';
                summary.key_findings.forEach(finding => {
                    html += `<li>${finding}</li>`;
                });
                html += '</ul>';
            }
            
            resultsContent.innerHTML = html;
            resultsSection.style.display = 'block';
        }
        
        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;
            
            const container = document.querySelector('.container');
            container.insertBefore(alertDiv, container.firstChild);
            
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
        
        function loadReports() {
            fetch('/reports')
            .then(response => response.json())
            .then(data => {
                const reportsList = document.getElementById('reportsList');
                
                if (data.success && data.reports.length > 0) {
                    let html = '';
                    data.reports.forEach(report => {
                        const date = new Date(report.date).toLocaleDateString('he-IL');
                        html += `
                            <div class="report-item">
                                <div>
                                    <strong>${report.original_file}</strong><br>
                                    <small>נוצר ב: ${date}</small>
                                </div>
                                <div class="report-actions">
                                    <button class="btn btn-primary" onclick="viewReport('${report.filename}')">
                                        צפה בדוח
                                    </button>
                                    <a href="/download/${report.filename}" class="btn btn-secondary">
                                        הורד JSON
                                    </a>
                                </div>
                            </div>
                        `;
                    });
                    reportsList.innerHTML = html;
                } else {
                    reportsList.innerHTML = '<p>אין דוחות זמינים</p>';
                }
            })
            .catch(error => {
                document.getElementById('reportsList').innerHTML = 
                    '<p>שגיאה בטעינת הדוחות</p>';
            });
        }
        
        function viewReport(filename) {
            fetch(`/report/${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Open report in new window
                    const reportWindow = window.open('', '_blank');
                    reportWindow.document.write(`
                        <html dir="rtl" lang="he">
                        <head>
                            <meta charset="UTF-8">
                            <title>דוח ניתוח - ${data.report.metadata.file_name}</title>
                            <style>
                                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f7fa; }
                                h1 { color: #667eea; }
                                .summary { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
                                .finding { background: #fff; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #667eea; }
                                .audio-player { 
                                    background: #e8f4f8; 
                                    padding: 15px; 
                                    border-radius: 8px; 
                                    margin: 10px 0; 
                                    border: 1px solid #dee2e6;
                                }
                                .audio-player audio { 
                                    width: 100%; 
                                    margin: 10px 0; 
                                    border-radius: 5px;
                                }
                            </style>
                        </head>
                        <body>
                            <h1>דוח ניתוח הקלטות גן ילדים</h1>
                            <div class="summary">
                                <h2>סיכום כללי</h2>
                                <p><strong>קובץ:</strong> ${data.report.metadata.file_name}</p>
                                <p><strong>תאריך ניתוח:</strong> ${data.report.metadata.analysis_date_hebrew}</p>
                                <p><strong>הערכה כללית:</strong> ${data.report.summary.overall_assessment}</p>
                                <p><strong>סה"כ אירועים:</strong> ${data.report.summary.total_incidents}</p>
                            </div>
                            
                            <h2>ממצאים מפורטים</h2>
                            ${generateDetailedFindingsHTML(data.report.detailed_findings, data.report.audio_clips || [])}
                            
                            <h2>המלצות</h2>
                            <ul>
                                ${data.report.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                            </ul>
                        </body>
                        </html>
                    `);
                }
            })
            .catch(error => {
                showAlert('שגיאה בטעינת הדוח', 'error');
            });
        }
        
        function generateDetailedFindingsHTML(findings, audioClips) {
            let html = '';
            
            // Find matching audio clip
            function findClip(incidentType, timestamp) {
                for (let clip of audioClips) {
                    if (clip.incident_type === incidentType) {
                        try {
                            const startTime = parseFloat(timestamp.split(' - ')[0].replace('s', ''));
                            if (Math.abs(clip.start_time - startTime) < 1.0) {
                                return clip;
                            }
                        } catch(e) {}
                    }
                }
                return null;
            }
            
            if (findings.emotional_analysis && findings.emotional_analysis.length > 0) {
                html += '<h3>ניתוח רגשי</h3>';
                findings.emotional_analysis.forEach(emotion => {
                    const clip = findClip('emotion', emotion.timestamp);
                    html += `<div class="finding">
                        <strong>${emotion.timestamp}</strong> - ${emotion.detected_emotion} (${emotion.severity})
                        ${clip ? `<div class="audio-player">
                            <p><strong>🎵 האזן לאירוע:</strong></p>
                            <audio controls>
                                <source src="/audio_clip/${clip.filename}" type="audio/wav">
                                הדפדפן שלך לא תומך בנגן אודיו
                            </audio>
                            <p><small>משך: ${clip.duration.toFixed(1)} שניות</small></p>
                        </div>` : ''}
                    </div>`;
                });
            }
            
            if (findings.violence_analysis && findings.violence_analysis.length > 0) {
                html += '<h3>ניתוח אלימות</h3>';
                findings.violence_analysis.forEach(violence => {
                    const clip = findClip('violence', violence.timestamp);
                    html += `<div class="finding">
                        <strong>${violence.timestamp}</strong> - ${violence.violence_types} (${violence.severity})
                        ${clip ? `<div class="audio-player">
                            <p><strong>🎵 האזן לאירוע:</strong></p>
                            <audio controls>
                                <source src="/audio_clip/${clip.filename}" type="audio/wav">
                                הדפדפן שלך לא תומך בנגן אודיו
                            </audio>
                            <p><small>משך: ${clip.duration.toFixed(1)} שניות</small></p>
                        </div>` : ''}
                    </div>`;
                });
            }
            
            if (findings.cry_analysis && findings.cry_analysis.length > 0) {
                html += '<h3>ניתוח בכי</h3>';
                findings.cry_analysis.forEach(cry => {
                    const clip = findClip('cry', cry.timestamp);
                    html += `<div class="finding">
                        <strong>${cry.timestamp}</strong> - ${cry.description}
                        ${clip ? `<div class="audio-player">
                            <p><strong>🎵 האזן לאירוע:</strong></p>
                            <audio controls>
                                <source src="/audio_clip/${clip.filename}" type="audio/wav">
                                הדפדפן שלך לא תומך בנגן אודיו
                            </audio>
                            <p><small>משך: ${clip.duration.toFixed(1)} שניות</small></p>
                        </div>` : ''}
                    </div>`;
                });
            }
            
            if (findings.neglect_analysis && findings.neglect_analysis.length > 0) {
                html += '<h3>ניתוח הזנחה</h3>';
                findings.neglect_analysis.forEach(neglect => {
                    const clip = findClip('neglect', neglect.timestamp);
                    html += `<div class="finding">
                        <strong>${neglect.timestamp}</strong> - ${neglect.description}
                        ${clip ? `<div class="audio-player">
                            <p><strong>🎵 האזן לאירוע:</strong></p>
                            <audio controls>
                                <source src="/audio_clip/${clip.filename}" type="audio/wav">
                                הדפדפן שלך לא תומך בנגן אודיו
                            </audio>
                            <p><small>משך: ${clip.duration.toFixed(1)} שניות</small></p>
                        </div>` : ''}
                    </div>`;
                });
            }
            
            return html;
        }
        
        
        // Load reports on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadReports();
        });
    </script>
</body>
</html>
    '''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

# =====================================================================
# WebSocket Live Monitoring (optional — requires flask-socketio)
# =====================================================================
try:
    from flask_socketio import SocketIO
    from live_monitor import register_socketio_events
    socketio = SocketIO(app, cors_allowed_origins="*")
    register_socketio_events(socketio, app)
    WEBSOCKET_AVAILABLE = True
except ImportError:
    socketio = None
    WEBSOCKET_AVAILABLE = False


if __name__ == '__main__':
    print("Initializing web application...")

    # Create templates directory if it doesn't exist
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)

    # Only create legacy template if the new dashboard doesn't exist
    template_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(template_path):
        create_html_template()

    print("Web application ready!")
    print(f"Open in browser: http://localhost:{config.web.port}")
    if WEBSOCKET_AVAILABLE:
        print("WebSocket live monitoring available at /ws namespace")
        socketio.run(app, debug=config.web.debug, host=config.web.host,
                     port=config.web.port)
    else:
        print("WebSocket not available (install flask-socketio for live monitoring)")
        app.run(debug=config.web.debug, host=config.web.host, port=config.web.port)
