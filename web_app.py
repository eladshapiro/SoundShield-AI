"""
Web Application for Kindergarten Recording Analyzer
אפליקציית ווב למערכת ניתוח הקלטות גן ילדים
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
            return jsonify({'error': 'לא נבחר קובץ'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'לא נבחר קובץ'}), 400
        
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
                analyzer = KindergartenRecordingAnalyzer()
            
            # Run analysis
            print(f"Starting analysis of {filename}...")
            results = analyzer.run_complete_analysis(filepath)
            
            # Return results
            return jsonify({
                'success': True,
                'message': 'ניתוח הושלם בהצלחה',
                'filename': filename,
                'results': {
                    'summary': results['report']['summary'],
                    'statistics': results['report']['statistics']
                }
            })
        
        else:
            return jsonify({'error': 'סוג קובץ לא נתמך'}), 400
    
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'שגיאה בניתוח: {str(e)}'}), 500

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
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 2em; }
            .results-grid { grid-template-columns: 1fr; }
            .report-item { flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>מערכת ניתוח הקלטות גן ילדים</h1>
            <p>Kindergarten Recording Analyzer</p>
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
            <p>מעבד את הקובץ... זה יכול לקחת מספר דקות</p>
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
        
        function handleFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Show loading
            loading.style.display = 'block';
            progressBar.style.display = 'block';
            resultsSection.style.display = 'none';
            
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
            }, 500);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(progressInterval);
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
                clearInterval(progressInterval);
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
                                body { font-family: Arial, sans-serif; margin: 20px; }
                                h1 { color: #667eea; }
                                .summary { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
                                .finding { background: #fff; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #667eea; }
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
                            ${generateDetailedFindingsHTML(data.report.detailed_findings)}
                            
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
        
        function generateDetailedFindingsHTML(findings) {
            let html = '';
            
            if (findings.emotional_analysis && findings.emotional_analysis.length > 0) {
                html += '<h3>ניתוח רגשי</h3>';
                findings.emotional_analysis.forEach(emotion => {
                    html += `<div class="finding">
                        <strong>${emotion.timestamp}</strong> - ${emotion.detected_emotion} (${emotion.severity})
                    </div>`;
                });
            }
            
            if (findings.violence_analysis && findings.violence_analysis.length > 0) {
                html += '<h3>ניתוח אלימות</h3>';
                findings.violence_analysis.forEach(violence => {
                    html += `<div class="finding">
                        <strong>${violence.timestamp}</strong> - ${violence.violence_types} (${violence.severity})
                    </div>`;
                });
            }
            
            if (findings.cry_analysis && findings.cry_analysis.length > 0) {
                html += '<h3>ניתוח בכי</h3>';
                findings.cry_analysis.forEach(cry => {
                    html += `<div class="finding">
                        <strong>${cry.timestamp}</strong> - ${cry.description}
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
