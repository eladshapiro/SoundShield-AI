"""
Web Application for Kindergarten Recording Analyzer
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import os
import json
import time
from werkzeug.utils import secure_filename
from main import KindergartenRecordingAnalyzer

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize analyzer
analyzer = None

# Progress tracking - use dictionary to support multiple concurrent analyses
# Key: filename, Value: progress dict
progress_tracking = {}
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
        'inappropriate_language': inappropriate_language,
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

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        language = request.form.get('language', 'en')  # Get language from form
        
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
            update_progress('completed', 'ניתוח הושלם בהצלחה! / Analysis completed successfully!', 7, filename=filename)
            
            # Clean up progress tracking after a delay (keep for 5 minutes for UI polling)
            # Note: In production, consider using a background task to clean up old entries
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
            
            return jsonify({
                'success': True,
                'message': 'Analysis completed successfully',
                'filename': filename,
                'results': {
                    'summary': results['report']['summary'],
                    'statistics': results['report']['statistics'],
                    'detailed_findings': results['report'].get('detailed_findings', {})
                },
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
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'analyzer_initialized': analyzer is not None
    })

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

if __name__ == '__main__':
    print("מאתחל אפליקציית ווב...")
    print("Initializing web application...")
    
    # Create templates directory if it doesn't exist
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create basic HTML template
    create_html_template()
    
    print("אפליקציית ווב מוכנה!")
    print("Web application ready!")
    print("פתח בדפדפן: http://localhost:5000")
    print("Open in browser: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
