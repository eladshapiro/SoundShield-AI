"""
Modern GUI Application for Kindergarten Recording Analyzer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console (only if not in GUI mode)
if sys.platform == 'win32' and not hasattr(sys, 'ps1'):
    try:
        import io
        if sys.stdout and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if sys.stderr and not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # Ignore if already wrapped or in GUI mode

from main import KindergartenRecordingAnalyzer


class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Language settings (default: English)
        self.language = 'en'  # 'en' or 'he'
        
        # Translations
        self.translations = {
            'en': {
                'title': 'Kindergarten Recording Analyzer',
                'subtitle': 'Upload an audio file to analyze kindergarten recordings',
                'select_file': 'Select Audio File',
                'no_file': 'No file selected',
                'browse': 'Browse Files',
                'analyze': 'Analyze Audio File',
                'progress': 'Analysis Progress',
                'ready': 'Ready to analyze',
                'results': 'Analysis Results',
                'language': 'Language / שפה',
                'select_lang': 'Select Language',
                'hebrew': 'Hebrew / עברית',
                'english': 'English / אנגלית',
                'init_analyzer': 'Initializing analyzer... Please wait',
                'analyzer_ready': 'Analyzer ready! Select a file to begin',
                'error_init': 'Error initializing analyzer',
                'error_file': 'Please select a valid audio file first.',
                'warning_init': 'Analyzer is still initializing. Please wait...',
                'step': 'Step',
                'analyzing': 'Analyzing audio file...',
                'detecting_emotions': 'Detecting emotions...',
                'detecting_cries': 'Detecting baby cries...',
                'detecting_violence': 'Detecting violence indicators...',
                'detecting_neglect': 'Detecting neglect patterns...',
                'advanced_analysis': 'Running advanced analysis...',
                'detecting_language': 'Detecting inappropriate language...',
                'generating_report': 'Generating report...',
                'complete': 'Analysis complete!',
                'error_analysis': 'Error during analysis',
                'welcome': 'WELCOME TO KINDERGARTEN RECORDING ANALYZER',
                'instructions': 'INSTRUCTIONS:',
                'instr1': "1. Select your preferred language",
                'instr2': "2. Click 'Browse Files' to select an audio file",
                'instr3': "3. Click 'Analyze Audio File' to start the analysis",
                'instr4': "4. Wait for the analysis to complete",
                'instr5': "5. Review the results below",
                'formats': 'SUPPORTED FORMATS:',
                'formats_list': 'WAV, MP3, M4A, FLAC, AAC, OGG',
                'time': 'ANALYSIS TIME:',
                'time_desc': 'Analysis typically takes 1-5 minutes depending on file size.\nPlease be patient and do not close the application.',
                'file_selected': 'File selected',
                'ready_analyze': 'Ready to analyze. Click ',
                'to_begin': " to begin.",
            },
            'he': {
                'title': 'מערכת ניתוח הקלטות גן ילדים',
                'subtitle': 'העלה קובץ אודיו לניתוח הקלטות גן ילדים',
                'select_file': 'בחירת קובץ אודיו',
                'no_file': 'לא נבחר קובץ',
                'browse': 'עיון בקבצים',
                'analyze': 'ניתוח קובץ אודיו',
                'progress': 'התקדמות הניתוח',
                'ready': 'מוכן לניתוח',
                'results': 'תוצאות הניתוח',
                'language': 'שפה / Language',
                'select_lang': 'בחר שפה',
                'hebrew': 'עברית / Hebrew',
                'english': 'אנגלית / English',
                'init_analyzer': 'מאתחל מנתח... אנא המתן',
                'analyzer_ready': 'מנתח מוכן! בחר קובץ להתחלה',
                'error_init': 'שגיאה באתחול מנתח',
                'error_file': 'אנא בחר קובץ אודיו תקין תחילה.',
                'warning_init': 'המנתח עדיין מאתחל. אנא המתן...',
                'step': 'שלב',
                'analyzing': 'מנתח קובץ אודיו...',
                'detecting_emotions': 'מזהה רגשות...',
                'detecting_cries': 'מזהה בכי תינוקות...',
                'detecting_violence': 'מזהה אינדיקטורים לאלימות...',
                'detecting_neglect': 'מזהה דפוסי הזנחה...',
                'advanced_analysis': 'מריץ ניתוח מתקדם...',
                'detecting_language': 'מזהה שפה לא הולמת...',
                'generating_report': 'יוצר דוח...',
                'complete': 'הניתוח הושלם!',
                'error_analysis': 'שגיאה במהלך הניתוח',
                'welcome': 'ברוכים הבאים למערכת ניתוח הקלטות גן ילדים',
                'instructions': 'הוראות:',
                'instr1': '1. בחר את שפת הממשק המועדפת עליך',
                'instr2': "2. לחץ על 'עיון בקבצים' כדי לבחור קובץ אודיו",
                'instr3': "3. לחץ על 'ניתוח קובץ אודיו' כדי להתחיל את הניתוח",
                'instr4': '4. המתן עד שהניתוח יושלם',
                'instr5': '5. סקור את התוצאות למטה',
                'formats': 'פורמטים נתמכים:',
                'formats_list': 'WAV, MP3, M4A, FLAC, AAC, OGG',
                'time': 'זמן ניתוח:',
                'time_desc': 'הניתוח בדרך כלל לוקח 1-5 דקות בהתאם לגודל הקובץ.\nאנא היה סבלני ואל תסגור את האפליקציה.',
                'file_selected': 'קובץ נבחר',
                'ready_analyze': 'מוכן לניתוח. לחץ על ',
                'to_begin': ' כדי להתחיל.',
            }
        }
        
        # Set minimum window size
        self.root.minsize(800, 600)
        
        # Make sure window appears on top initially
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        
        # Center window on screen
        self.center_window()
        
        # Initialize analyzer (will be done in background)
        self.analyzer = None
        self.current_file = None
        self.analysis_results = None
        
        # Configure style
        self.setup_styles()
        
        # Create UI
        self.create_widgets()
        
        # Initialize analyzer in background thread
        self.init_analyzer()
    
    def t(self, key: str) -> str:
        """Get translation for current language"""
        return self.translations[self.language].get(key, key)
    
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 20, 'bold'),
                       background='#f0f0f0',
                       foreground='#2c3e50')
        
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 11),
                       background='#f0f0f0',
                       foreground='#7f8c8d')
        
        style.configure('Header.TLabel',
                       font=('Segoe UI', 14, 'bold'),
                       background='#f0f0f0',
                       foreground='#34495e')
        
        style.configure('Modern.TButton',
                       font=('Segoe UI', 11, 'bold'),
                       padding=10)
        
        # Note: Progress bar will use default style for compatibility
    
    def change_language(self, lang: str):
        """Change interface language"""
        self.language = lang
        self.update_ui_language()
        # Re-initialize analyzer with new language if already initialized
        if self.analyzer:
            # Note: This would require modifying main.py to accept language parameter
            pass
    
    def update_ui_language(self):
        """Update all UI text to current language"""
        self.root.title(self.t('title'))
        self.title_label.config(text=f"🎵 {self.t('title')}")
        self.subtitle_label.config(text=self.t('subtitle'))
        self.select_file_label.config(text=f"📁 {self.t('select_file')}")
        self.file_label.config(text=self.t('no_file'))
        self.select_btn.config(text=self.t('browse'))
        self.analyze_btn.config(text=f"🔍 {self.t('analyze')}")
        self.progress_label_header.config(text=f"📊 {self.t('progress')}")
        self.status_label.config(text=self.t('ready'))
        self.results_label_header.config(text=f"📋 {self.t('results')}")
        self.language_label.config(text=f"🌐 {self.t('language')}")
        
        # Update welcome message
        self.update_welcome_message()
    
    def update_welcome_message(self):
        """Update welcome message based on current language"""
        welcome_msg = "=" * 70 + "\n"
        welcome_msg += f"{self.t('welcome')}\n"
        welcome_msg += "=" * 70 + "\n\n"
        welcome_msg += f"📋 {self.t('instructions')}\n"
        welcome_msg += f"{self.t('instr1')}\n"
        welcome_msg += f"{self.t('instr2')}\n"
        welcome_msg += f"{self.t('instr3')}\n"
        welcome_msg += f"{self.t('instr4')}\n"
        welcome_msg += f"{self.t('instr5')}\n\n"
        welcome_msg += f"📁 {self.t('formats')}\n"
        welcome_msg += f"   {self.t('formats_list')}\n\n"
        welcome_msg += f"⏱️ {self.t('time')}\n"
        welcome_msg += f"   {self.t('time_desc')}\n"
        self.update_results(welcome_msg)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#f0f0f0')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Language selection (top right)
        lang_frame = tk.Frame(header_frame, bg='#f0f0f0')
        lang_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.language_label = ttk.Label(lang_frame,
                                        text=f"🌐 {self.t('language')}",
                                        style='Subtitle.TLabel')
        self.language_label.pack(side=tk.LEFT)
        
        lang_buttons_frame = tk.Frame(lang_frame, bg='#f0f0f0')
        lang_buttons_frame.pack(side=tk.RIGHT)
        
        self.lang_var = tk.StringVar(value=self.language)
        lang_en_btn = ttk.Radiobutton(lang_buttons_frame,
                                      text=self.t('english'),
                                      variable=self.lang_var,
                                      value='en',
                                      command=lambda: self.change_language('en'))
        lang_en_btn.pack(side=tk.LEFT, padx=5)
        
        lang_he_btn = ttk.Radiobutton(lang_buttons_frame,
                                      text=self.t('hebrew'),
                                      variable=self.lang_var,
                                      value='he',
                                      command=lambda: self.change_language('he'))
        lang_he_btn.pack(side=tk.LEFT, padx=5)
        
        self.title_label = ttk.Label(header_frame, 
                               text=f"🎵 {self.t('title')}",
                               style='Title.TLabel')
        self.title_label.pack()
        
        self.subtitle_label = ttk.Label(header_frame,
                                  text=self.t('subtitle'),
                                  style='Subtitle.TLabel')
        self.subtitle_label.pack(pady=(5, 0))
        
        # Upload section
        upload_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=1)
        upload_frame.pack(fill=tk.X, pady=(0, 20))
        
        upload_inner = tk.Frame(upload_frame, bg='white', padx=30, pady=30)
        upload_inner.pack(fill=tk.BOTH, expand=True)
        
        self.select_file_label = ttk.Label(upload_inner,
                 text=f"📁 {self.t('select_file')}",
                 style='Header.TLabel',
                 background='white')
        self.select_file_label.pack(anchor=tk.W, pady=(0, 15))
        
        # File selection frame
        file_frame = tk.Frame(upload_inner, bg='white')
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.file_label = tk.Label(file_frame,
                                   text=self.t('no_file'),
                                   font=('Segoe UI', 10),
                                   bg='white',
                                   fg='#7f8c8d',
                                   anchor=tk.W)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.select_btn = ttk.Button(file_frame,
                                     text=self.t('browse'),
                                     command=self.select_file,
                                     style='Modern.TButton')
        self.select_btn.pack(side=tk.RIGHT)
        
        # Analyze button
        self.analyze_btn = ttk.Button(upload_inner,
                                     text=f"🔍 {self.t('analyze')}",
                                     command=self.start_analysis,
                                     style='Modern.TButton',
                                     state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Progress section
        progress_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=1)
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        progress_inner = tk.Frame(progress_frame, bg='white', padx=30, pady=20)
        progress_inner.pack(fill=tk.BOTH, expand=True)
        
        self.progress_label_header = ttk.Label(progress_inner,
                 text=f"📊 {self.t('progress')}",
                 style='Header.TLabel',
                 background='white')
        self.progress_label_header.pack(anchor=tk.W, pady=(0, 15))
        
        # Status label
        self.status_label = tk.Label(progress_inner,
                                     text=self.t('ready'),
                                     font=('Segoe UI', 10),
                                     bg='white',
                                     fg='#7f8c8d',
                                     anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar (using default style for compatibility)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_inner,
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=400,
                                           mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Progress percentage
        self.progress_label = tk.Label(progress_inner,
                                       text="0%",
                                       font=('Segoe UI', 9),
                                       bg='white',
                                       fg='#3498db',
                                       anchor=tk.E)
        self.progress_label.pack(fill=tk.X)
        
        # Results section
        results_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=1)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_inner = tk.Frame(results_frame, bg='white', padx=30, pady=20)
        results_inner.pack(fill=tk.BOTH, expand=True)
        
        self.results_label_header = ttk.Label(results_inner,
                 text=f"📋 {self.t('results')}",
                 style='Header.TLabel',
                 background='white')
        self.results_label_header.pack(anchor=tk.W, pady=(0, 15))
        
        # Results text area with scrollbar
        results_container = tk.Frame(results_inner, bg='white')
        results_container.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_container,
                                                      wrap=tk.WORD,
                                                      font=('Segoe UI', 10),
                                                      bg='#fafafa',
                                                      fg='#2c3e50',
                                                      relief=tk.FLAT,
                                                      padx=15,
                                                      pady=15,
                                                      state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.update_welcome_message()
    
    def init_analyzer(self):
        """Initialize analyzer in background thread"""
        def init_thread():
            self.update_status(self.t('init_analyzer'))
            self.update_progress(5)
            try:
                # Pass language to analyzer
                self.analyzer = KindergartenRecordingAnalyzer(language=self.language)
                self.update_status(self.t('analyzer_ready'))
                self.update_progress(0)
            except Exception as e:
                self.update_status(f"{self.t('error_init')}: {str(e)}")
                messagebox.showerror(self.t('error_init'), 
                                   f"Failed to initialize analyzer:\n{str(e)}")
        
        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()
    
    def select_file(self):
        """Open file dialog to select audio file"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.m4a *.flac *.aac *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        title = "Select Audio File" if self.language == 'en' else "בחר קובץ אודיו"
        filename = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        
        if filename:
            self.current_file = filename
            file_name = os.path.basename(filename)
            self.file_label.config(text=f"{self.t('file_selected')}: {file_name}", fg='#27ae60')
            self.analyze_btn.config(state=tk.NORMAL)
            ready_msg = f"{self.t('file_selected')}: {file_name}\n\n{self.t('ready_analyze')}'{self.t('analyze')}'{self.t('to_begin')}"
            self.update_results(ready_msg)
    
    def start_analysis(self):
        """Start analysis in background thread"""
        if not self.current_file or not os.path.exists(self.current_file):
            messagebox.showerror("Error", self.t('error_file'))
            return
        
        if not self.analyzer:
            messagebox.showwarning("Warning", self.t('warning_init'))
            return
        
        # Disable button during analysis
        self.analyze_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)
        
        # Start analysis in background thread
        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()
    
    def run_analysis(self):
        """Run the analysis (called in background thread)"""
        try:
            file_path = self.current_file
            
            # Step 1: Basic audio analysis
            step1_msg = f"{self.t('step')} 1/7: {self.t('analyzing')}"
            self.update_status(step1_msg)
            self.update_progress(5)
            self.root.after(0, lambda: self.update_results(f"🔍 {self.t('step')} 1: {self.t('analyzing')}\n"
                                                           "   - Loading audio file\n"
                                                           "   - Extracting audio features\n"
                                                           "   - Identifying segments\n"))
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            audio_analysis = self.analyzer.audio_analyzer.analyze_audio_file(file_path)
            self.update_progress(15)
            
            # Step 2: Emotion detection
            step2_msg = f"{self.t('step')} 2/7: {self.t('detecting_emotions')}"
            self.update_status(step2_msg)
            self.update_progress(20)
            self.root.after(0, lambda: self.update_results(f"😊 {self.t('step')} 2: {self.t('detecting_emotions')}\n"
                                                           "   - Analyzing emotional patterns\n"
                                                           "   - Identifying concerning emotions\n"))
            
            emotion_results = self.analyzer.emotion_detector.analyze_segment_emotions(
                audio_analysis['segments'], 
                audio_analysis['sample_rate']
            )
            concerning_emotions = self.analyzer.emotion_detector.detect_concerning_emotions(emotion_results)
            self.update_progress(35)
            
            # Step 3: Cry detection
            step3_msg = f"{self.t('step')} 3/7: {self.t('detecting_cries')}"
            self.update_status(step3_msg)
            self.update_progress(40)
            self.root.after(0, lambda: self.update_results(f"👶 {self.t('step')} 3: {self.t('detecting_cries')}\n"
                                                           "   - Identifying cry patterns\n"
                                                           "   - Checking for responses\n"))
            
            audio, sr = self.analyzer.audio_analyzer.load_audio(file_path)
            cry_segments = self.analyzer.cry_detector.detect_cry_segments(audio, sr)
            cry_with_responses = self.analyzer.cry_detector.detect_response_to_cry(audio, sr, cry_segments)
            self.update_progress(55)
            
            # Step 4: Violence detection
            step4_msg = f"{self.t('step')} 4/7: {self.t('detecting_violence')}"
            self.update_status(step4_msg)
            self.update_progress(60)
            self.root.after(0, lambda: self.update_results(f"⚠️ {self.t('step')} 4: {self.t('detecting_violence')}\n"
                                                           "   - Analyzing aggressive patterns\n"
                                                           "   - Identifying potential threats\n"))
            
            violence_segments = self.analyzer.violence_detector.detect_violence_segments(audio, sr)
            self.update_progress(70)
            
            # Step 5: Neglect detection
            step5_msg = f"{self.t('step')} 5/7: {self.t('detecting_neglect')}"
            self.update_status(step5_msg)
            self.update_progress(75)
            self.root.after(0, lambda: self.update_results(f"🚨 {self.t('step')} 5: {self.t('detecting_neglect')}\n"
                                                           "   - Analyzing response patterns\n"
                                                           "   - Identifying neglect indicators\n"))
            
            neglect_analysis = self.analyzer.neglect_detector.detect_neglect_patterns(
                audio, sr, cry_segments, violence_segments
            )
            self.update_progress(80)
            
            # Step 6: Advanced analysis (if available)
            advanced_analysis = {}
            if self.analyzer.advanced_analyzer and self.analyzer.advanced_analyzer.models_loaded:
                step6_msg = f"{self.t('step')} 6/7: {self.t('advanced_analysis')}"
                self.update_status(step6_msg)
                self.update_progress(82)
                self.root.after(0, lambda: self.update_results(f"🤖 {self.t('step')} 6: {self.t('advanced_analysis')}\n"
                                                               "   - Using ML models\n"
                                                               "   - Deep analysis in progress\n"))
                try:
                    # Pass language to advanced analyzer
                    advanced_analysis = self.analyzer.advanced_analyzer.comprehensive_analysis(
                        file_path, language=self.language
                    )
                    self.update_progress(88)
                except Exception as e:
                    print(f"Advanced analysis error: {e}")
            
            # Step 7: Language detection
            inappropriate_language = {}
            if self.analyzer.language_detector:
                step7_msg = f"{self.t('step')} 7/7: {self.t('detecting_language')}"
                self.update_status(step7_msg)
                self.update_progress(90)
                self.root.after(0, lambda: self.update_results(f"🔤 {self.t('step')} 7: {self.t('detecting_language')}\n"
                                                               "   - Transcribing audio\n"
                                                               "   - Checking for inappropriate words\n"))
                try:
                    # Pass language to language detector
                    inappropriate_language = self.analyzer.language_detector.analyze_with_whisper(
                        file_path, language=self.language
                    )
                    self.update_progress(95)
                except Exception as e:
                    print(f"Language detection error: {e}")
            
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
                'language': self.language
            }
            
            # Generate report
            self.update_status(self.t('generating_report'))
            self.update_progress(97)
            self.root.after(0, lambda: self.update_results(f"📄 {self.t('generating_report')}\n"
                                                           "   - Compiling findings\n"
                                                           "   - Creating recommendations\n"))
            
            report = self.analyzer.generate_report(analysis_results)
            analysis_results['report'] = report
            self.analysis_results = analysis_results
            
            # Display results
            self.update_progress(100)
            self.update_status(self.t('complete'))
            self.root.after(0, lambda: self.display_results(analysis_results))
            
        except Exception as e:
            error_msg = f"{self.t('error_analysis')}: {str(e)}"
            self.update_status(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
            self.root.after(0, lambda: self.update_results(f"❌ {error_msg}\n\nPlease try again with a different file."))
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.select_btn.config(state=tk.NORMAL))
    
    def display_results(self, results: Dict):
        """Display analysis results in the results text area"""
        report = results.get('report', {})
        summary = report.get('summary', {})
        stats = report.get('statistics', {})
        
        # Clear and enable text widget
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Build results text based on language
        if self.language == 'he':
            results_text = self._build_results_hebrew(results, report, summary, stats)
        else:
            results_text = self._build_results_english(results, report, summary, stats)
        
        # Insert text
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state=tk.DISABLED)
        
        # Scroll to top
        self.results_text.see(1.0)
    
    def _build_results_english(self, results: Dict, report: Dict, summary: Dict, stats: Dict) -> str:
        """Build results text in English"""
        results_text = "=" * 70 + "\n"
        results_text += "ANALYSIS RESULTS\n"
        results_text += "=" * 70 + "\n\n"
        
        # Overall Assessment
        results_text += "📊 OVERALL ASSESSMENT\n"
        results_text += "-" * 70 + "\n"
        results_text += f"Assessment: {summary.get('overall_assessment', 'N/A')}\n"
        results_text += f"Risk Level: {summary.get('risk_level', 'N/A')}\n"
        results_text += f"Total Incidents: {summary.get('total_incidents', 0)}\n"
        results_text += f"Critical Incidents: {summary.get('critical_incidents', 0)}\n\n"
        
        # Statistics
        results_text += "📈 STATISTICS\n"
        results_text += "-" * 70 + "\n"
        if stats:
            results_text += f"Recording Duration: {stats.get('audio_duration_minutes', 0):.2f} minutes\n"
            results_text += f"Total Segments Analyzed: {stats.get('total_segments', 0)}\n"
            results_text += f"Silent Segments: {stats.get('silent_segments_count', 0)}\n"
            results_text += f"Loud Segments: {stats.get('loud_segments_count', 0)}\n\n"
        
        # Key Findings
        key_findings = summary.get('key_findings', [])
        if key_findings:
            results_text += "🔍 KEY FINDINGS\n"
            results_text += "-" * 70 + "\n"
            for i, finding in enumerate(key_findings, 1):
                results_text += f"{i}. {finding}\n"
            results_text += "\n"
        
        # Detailed Findings
        detailed = report.get('detailed_findings', {})
        
        # Emotion Analysis
        if detailed.get('emotional_analysis'):
            results_text += "😊 EMOTION ANALYSIS\n"
            results_text += "-" * 70 + "\n"
            for emotion in detailed['emotional_analysis'][:10]:
                results_text += f"  • {emotion.get('timestamp', 'N/A')}: "
                results_text += f"{emotion.get('detected_emotion', 'N/A')} "
                results_text += f"({emotion.get('severity', 'N/A')})\n"
            if len(detailed['emotional_analysis']) > 10:
                results_text += f"  ... and {len(detailed['emotional_analysis']) - 10} more\n"
            results_text += "\n"
        
        # Cry Analysis
        if detailed.get('cry_analysis'):
            results_text += "👶 CRY ANALYSIS\n"
            results_text += "-" * 70 + "\n"
            for cry in detailed['cry_analysis'][:10]:
                results_text += f"  • {cry.get('timestamp', 'N/A')}: "
                results_text += f"{cry.get('description', 'N/A')}\n"
            if len(detailed['cry_analysis']) > 10:
                results_text += f"  ... and {len(detailed['cry_analysis']) - 10} more\n"
            results_text += "\n"
        
        # Violence Analysis
        if detailed.get('violence_analysis'):
            results_text += "⚠️ VIOLENCE ANALYSIS\n"
            results_text += "-" * 70 + "\n"
            for violence in detailed['violence_analysis'][:10]:
                results_text += f"  • {violence.get('timestamp', 'N/A')}: "
                results_text += f"{violence.get('violence_types', 'N/A')} "
                results_text += f"({violence.get('severity', 'N/A')})\n"
            if len(detailed['violence_analysis']) > 10:
                results_text += f"  ... and {len(detailed['violence_analysis']) - 10} more\n"
            results_text += "\n"
        
        # Neglect Analysis
        if detailed.get('neglect_analysis'):
            results_text += "🚨 NEGLECT ANALYSIS\n"
            results_text += "-" * 70 + "\n"
            for neglect in detailed['neglect_analysis'][:10]:
                results_text += f"  • {neglect.get('timestamp', 'N/A')}: "
                results_text += f"{neglect.get('description', 'N/A')}\n"
            if len(detailed['neglect_analysis']) > 10:
                results_text += f"  ... and {len(detailed['neglect_analysis']) - 10} more\n"
            results_text += "\n"
        
        # Inappropriate Language
        inappropriate = results.get('inappropriate_language', {})
        if inappropriate.get('detected_inappropriate_words', 0) > 0:
            results_text += "🔤 INAPPROPRIATE LANGUAGE DETECTED\n"
            results_text += "-" * 70 + "\n"
            results_text += f"Detected Words: {inappropriate.get('detected_inappropriate_words', 0)}\n"
            if inappropriate.get('inappropriate_words'):
                results_text += "Words: " + ", ".join(inappropriate['inappropriate_words'][:20]) + "\n"
            results_text += "\n"
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            results_text += "💡 RECOMMENDATIONS\n"
            results_text += "-" * 70 + "\n"
            for i, rec in enumerate(recommendations, 1):
                results_text += f"{i}. {rec}\n"
            results_text += "\n"
        
        # Report location
        results_text += "=" * 70 + "\n"
        results_text += "📄 Full report has been saved to the 'reports' directory.\n"
        results_text += "=" * 70 + "\n"
        
        return results_text
    
    def _build_results_hebrew(self, results: Dict, report: Dict, summary: Dict, stats: Dict) -> str:
        """Build results text in Hebrew"""
        results_text = "=" * 70 + "\n"
        results_text += "תוצאות הניתוח\n"
        results_text += "=" * 70 + "\n\n"
        
        # Overall Assessment
        results_text += "📊 הערכה כללית\n"
        results_text += "-" * 70 + "\n"
        results_text += f"הערכה: {summary.get('overall_assessment', 'לא זמין')}\n"
        results_text += f"רמת סיכון: {summary.get('risk_level', 'לא זמין')}\n"
        results_text += f"סה\"כ אירועים: {summary.get('total_incidents', 0)}\n"
        results_text += f"אירועים קריטיים: {summary.get('critical_incidents', 0)}\n\n"
        
        # Statistics
        results_text += "📈 סטטיסטיקות\n"
        results_text += "-" * 70 + "\n"
        if stats:
            results_text += f"משך הקלטה: {stats.get('audio_duration_minutes', 0):.2f} דקות\n"
            results_text += f"סה\"כ קטעים שניתחו: {stats.get('total_segments', 0)}\n"
            results_text += f"קטעים שקטים: {stats.get('silent_segments_count', 0)}\n"
            results_text += f"קטעים רועשים: {stats.get('loud_segments_count', 0)}\n\n"
        
        # Key Findings
        key_findings = summary.get('key_findings', [])
        if key_findings:
            results_text += "🔍 ממצאים עיקריים\n"
            results_text += "-" * 70 + "\n"
            for i, finding in enumerate(key_findings, 1):
                results_text += f"{i}. {finding}\n"
            results_text += "\n"
        
        # Detailed Findings
        detailed = report.get('detailed_findings', {})
        
        # Emotion Analysis
        if detailed.get('emotional_analysis'):
            results_text += "😊 ניתוח רגשי\n"
            results_text += "-" * 70 + "\n"
            for emotion in detailed['emotional_analysis'][:10]:
                results_text += f"  • {emotion.get('timestamp', 'לא זמין')}: "
                results_text += f"{emotion.get('detected_emotion', 'לא זמין')} "
                results_text += f"({emotion.get('severity', 'לא זמין')})\n"
            if len(detailed['emotional_analysis']) > 10:
                results_text += f"  ... ועוד {len(detailed['emotional_analysis']) - 10}\n"
            results_text += "\n"
        
        # Cry Analysis
        if detailed.get('cry_analysis'):
            results_text += "👶 ניתוח בכי\n"
            results_text += "-" * 70 + "\n"
            for cry in detailed['cry_analysis'][:10]:
                results_text += f"  • {cry.get('timestamp', 'לא זמין')}: "
                results_text += f"{cry.get('description', 'לא זמין')}\n"
            if len(detailed['cry_analysis']) > 10:
                results_text += f"  ... ועוד {len(detailed['cry_analysis']) - 10}\n"
            results_text += "\n"
        
        # Violence Analysis
        if detailed.get('violence_analysis'):
            results_text += "⚠️ ניתוח אלימות\n"
            results_text += "-" * 70 + "\n"
            for violence in detailed['violence_analysis'][:10]:
                results_text += f"  • {violence.get('timestamp', 'לא זמין')}: "
                results_text += f"{violence.get('violence_types', 'לא זמין')} "
                results_text += f"({violence.get('severity', 'לא זמין')})\n"
            if len(detailed['violence_analysis']) > 10:
                results_text += f"  ... ועוד {len(detailed['violence_analysis']) - 10}\n"
            results_text += "\n"
        
        # Neglect Analysis
        if detailed.get('neglect_analysis'):
            results_text += "🚨 ניתוח הזנחה\n"
            results_text += "-" * 70 + "\n"
            for neglect in detailed['neglect_analysis'][:10]:
                results_text += f"  • {neglect.get('timestamp', 'לא זמין')}: "
                results_text += f"{neglect.get('description', 'לא זמין')}\n"
            if len(detailed['neglect_analysis']) > 10:
                results_text += f"  ... ועוד {len(detailed['neglect_analysis']) - 10}\n"
            results_text += "\n"
        
        # Inappropriate Language
        inappropriate = results.get('inappropriate_language', {})
        if inappropriate.get('detected_inappropriate_words', 0) > 0:
            results_text += "🔤 זוהי שפה לא הולמת\n"
            results_text += "-" * 70 + "\n"
            results_text += f"מילים שזוהו: {inappropriate.get('detected_inappropriate_words', 0)}\n"
            if inappropriate.get('inappropriate_words'):
                results_text += "מילים: " + ", ".join(inappropriate['inappropriate_words'][:20]) + "\n"
            results_text += "\n"
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            results_text += "💡 המלצות\n"
            results_text += "-" * 70 + "\n"
            for i, rec in enumerate(recommendations, 1):
                results_text += f"{i}. {rec}\n"
            results_text += "\n"
        
        # Report location
        results_text += "=" * 70 + "\n"
        results_text += "📄 דוח מלא נשמר בתיקיית 'reports'.\n"
        results_text += "=" * 70 + "\n"
        
        return results_text
    
    def update_status(self, message: str):
        """Update status label (thread-safe)"""
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def update_progress(self, value: float):
        """Update progress bar (thread-safe)"""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(text=f"{int(value)}%"))
    
    def update_results(self, message: str):
        """Update results text area (thread-safe)"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, message)
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(1.0)


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = ModernGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
