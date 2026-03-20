"""
Main Application for Kindergarten Recording Analyzer

This module serves as the main orchestrator for the SoundShield-AI system,
coordinating all analysis modules to detect inappropriate behavior in
kindergarten audio recordings.
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, Callable
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('soundshield.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import our custom modules
try:
    from audio_analyzer import AudioAnalyzer
    from emotion_detector import EmotionDetector
    from cry_detector import CryDetector
    from violence_detector import ViolenceDetector
    from neglect_detector import NeglectDetector
    from report_generator import ReportGenerator
except ImportError as e:
    logger.critical(f"Failed to import core modules: {e}")
    sys.exit(1)

# Import advanced models (optional)
try:
    from advanced_analyzer import AdvancedAnalyzer
    ADVANCED_MODELS_AVAILABLE = True
    logger.info("Advanced models module available")
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logger.warning("Advanced models not available (optional)")

# Import inappropriate language detector
try:
    from inappropriate_language_detector import InappropriateLanguageDetector
    LANGUAGE_DETECTOR_AVAILABLE = True
    logger.info("Inappropriate language detector available")
except ImportError:
    LANGUAGE_DETECTOR_AVAILABLE = False
    logger.warning("Inappropriate language detector not available (optional)")

# Import speaker diarizer
try:
    from speaker_diarizer import SpeakerDiarizer
    DIARIZER_AVAILABLE = True
    logger.info("Speaker diarizer available")
except ImportError:
    DIARIZER_AVAILABLE = False
    logger.warning("Speaker diarizer not available (optional)")

# Constants
SUPPORTED_LANGUAGES = ['en', 'he']
SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']
MAX_AUDIO_LENGTH_SECONDS = 3600  # 1 hour
MIN_AUDIO_LENGTH_SECONDS = 1

class AudioAnalysisError(Exception):
    """Base exception for audio analysis errors."""
    pass


class InvalidAudioFormatError(AudioAnalysisError):
    """Raised when audio format is not supported."""
    pass


class AudioFileTooLongError(AudioAnalysisError):
    """Raised when audio file exceeds maximum length."""
    pass


class KindergartenRecordingAnalyzer:
    """Main analyzer coordinating all detection modules.

    This class orchestrates comprehensive analysis of kindergarten audio
    recordings to detect inappropriate staff behavior including emotional
    abuse, neglect, violence, and inappropriate language.

    Args:
        language: Language code for analysis ('en' for English, 'he' for Hebrew)
        use_advanced: When True (default), eagerly loads ML models as primary detectors

    Attributes:
        language: The language used for analysis
        audio_analyzer: Basic audio analysis module
        emotion_detector: Emotion detection module
        cry_detector: Baby cry detection module
        violence_detector: Violence detection module
        neglect_detector: Neglect detection module
        advanced_analyzer: Advanced ML models (optional)
        language_detector: Inappropriate language detector (optional)
        report_generator: Report generation module

    Example:
        >>> analyzer = KindergartenRecordingAnalyzer(language='en')
        >>> results = analyzer.run_complete_analysis('recording.wav')
        >>> print(results['report']['summary']['risk_level'])
    """

    def __init__(self, language: str = 'en', use_advanced: bool = True):
        """Initialize the main analyzer application."""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        self.language = language
        self.use_advanced = use_advanced
        logger.info(f"Initializing SoundShield-AI Analyzer (language: {language}, use_advanced: {use_advanced})")

        try:
            # Initialize all analysis modules
            self.audio_analyzer = AudioAnalyzer()
            self.emotion_detector = EmotionDetector()
            self.cry_detector = CryDetector()
            self.violence_detector = ViolenceDetector()
            self.neglect_detector = NeglectDetector()
            self.report_generator = ReportGenerator()
            logger.info("Core modules initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize core modules: {e}")
            raise AudioAnalysisError(f"Initialization failed: {e}")

        # Initialize advanced analyzer - eagerly loaded as default when use_advanced=True
        self.advanced_analyzer = None
        if ADVANCED_MODELS_AVAILABLE and use_advanced:
            try:
                self.advanced_analyzer = AdvancedAnalyzer(
                    use_whisper=True,
                    use_transformer_emotion=True
                )
                self.advanced_analyzer.load_models()
                if self.advanced_analyzer.models_loaded:
                    logger.info(
                        f"Advanced models loaded (whisper={self.advanced_analyzer.whisper_loaded}, "
                        f"hubert={self.advanced_analyzer.hubert_loaded})"
                    )
            except Exception as e:
                logger.warning(f"Error loading advanced models: {e}")

        # Initialize speaker diarizer
        self.speaker_diarizer = None
        if DIARIZER_AVAILABLE and use_advanced:
            try:
                self.speaker_diarizer = SpeakerDiarizer()
                logger.info("Speaker diarizer initialized")
            except Exception as e:
                logger.warning(f"Error initializing speaker diarizer: {e}")

        # Initialize inappropriate language detector
        self.language_detector = None
        if LANGUAGE_DETECTOR_AVAILABLE:
            try:
                self.language_detector = InappropriateLanguageDetector()
                logger.info("Inappropriate language detector initialized")
            except Exception as e:
                logger.warning(f"Error initializing language detector: {e}")

        logger.info("All modules initialized successfully")
    
    def _validate_audio_file(self, file_path: str) -> None:
        """Validate audio file exists and has supported format.
        
        Args:
            file_path: Path to audio file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            InvalidAudioFormatError: If format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise InvalidAudioFormatError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size (approximate duration check)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 500:  # 500MB limit
            logger.warning(f"Large audio file detected: {file_size_mb:.1f}MB")
    
    def analyze_audio_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict:
        """Perform complete analysis of an audio file.
        
        Args:
            file_path: Path to audio file
            language: Language for analysis ('en' or 'he'). If None, uses instance language.
            progress_callback: Optional callback for progress updates.
                             Called with (current_step, total_steps, message)
                             
        Returns:
            Dictionary containing all analysis results
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            InvalidAudioFormatError: If audio format is not supported
            AudioAnalysisError: If analysis fails
            
        Example:
            >>> def progress(current, total, msg):
            ...     print(f"[{current}/{total}] {msg}")
            >>> analyzer = KindergartenRecordingAnalyzer()
            >>> results = analyzer.analyze_audio_file('recording.wav', progress_callback=progress)
        """
        if language is None:
            language = self.language
        
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
            
        logger.info(f"Starting analysis of file: {file_path}")
        
        # Validate file
        self._validate_audio_file(file_path)
        
        total_steps = 7
        current_step = 0
        
        def update_progress(message: str):
            nonlocal current_step
            current_step += 1
            logger.info(f"Step {current_step}/{total_steps}: {message}")
            if progress_callback:
                progress_callback(current_step, total_steps, message)
        
        try:
            # Step 1: Basic audio analysis
            update_progress("Basic audio analysis")
            audio_analysis = self.audio_analyzer.analyze_audio_file(file_path)

            # Validate audio duration
            duration = audio_analysis.get('duration', 0)
            if duration < MIN_AUDIO_LENGTH_SECONDS:
                raise AudioAnalysisError(
                    f"Audio file too short: {duration:.1f}s "
                    f"(minimum: {MIN_AUDIO_LENGTH_SECONDS}s)"
                )
            if duration > MAX_AUDIO_LENGTH_SECONDS:
                raise AudioFileTooLongError(
                    f"Audio file too long: {duration:.1f}s "
                    f"(maximum: {MAX_AUDIO_LENGTH_SECONDS}s)"
                )

            # Step 2: Emotion detection (HuBERT primary, heuristic fallback)
            update_progress("Emotion detection")
            heuristic_emotions = self.emotion_detector.analyze_segment_emotions(
                audio_analysis['segments'],
                audio_analysis['sample_rate']
            )
            heuristic_concerning = self.emotion_detector.detect_concerning_emotions(
                heuristic_emotions
            )

            # Use HuBERT as PRIMARY when available, merge with heuristics
            advanced_emotions = []
            hubert_used = False
            if (self.advanced_analyzer and
                    hasattr(self.advanced_analyzer, 'hubert_loaded') and
                    self.advanced_analyzer.hubert_loaded):
                try:
                    advanced_emotions = self.advanced_analyzer.detect_concerning_emotions_advanced(
                        file_path
                    )
                    hubert_used = True
                    logger.info(f"HuBERT detected {len(advanced_emotions)} concerning segments")
                except Exception as e:
                    logger.warning(f"HuBERT emotion detection failed, using heuristics: {e}")

            if hubert_used and advanced_emotions:
                concerning_emotions = self.emotion_detector.merge_with_advanced_results(
                    heuristic_concerning, advanced_emotions
                )
            else:
                concerning_emotions = heuristic_concerning
                for e in concerning_emotions:
                    e['ml_backed'] = False

            emotion_results = heuristic_emotions

            # Step 3: Cry detection
            update_progress("Baby cry detection")
            audio, sr = self.audio_analyzer.load_audio(file_path)
            cry_segments = self.cry_detector.detect_cry_segments(audio, sr)
            cry_with_responses = self.cry_detector.detect_response_to_cry(
                audio, sr, cry_segments
            )

            # Step 4: Violence detection
            update_progress("Violence detection")
            violence_segments = self.violence_detector.detect_violence_segments(audio, sr)

            # Step 5: Neglect detection
            update_progress("Neglect detection")
            neglect_analysis = self.neglect_detector.detect_neglect_patterns(
                audio, sr, cry_segments, violence_segments
            )

            # Step 6: Advanced analysis with ML models (Whisper + comprehensive)
            advanced_analysis = {}
            if self.advanced_analyzer and self.advanced_analyzer.models_loaded:
                update_progress("Advanced ML analysis")
                try:
                    advanced_analysis = self.advanced_analyzer.comprehensive_analysis(
                        file_path, language=language
                    )
                    logger.info("Advanced analysis completed successfully")
                except Exception as e:
                    logger.warning(f"Error in advanced analysis: {e}")

            # Speaker diarization (runs once, results available to all)
            diarization_results = {}
            if self.speaker_diarizer:
                try:
                    speaker_segments = self.speaker_diarizer.get_speaker_segments(file_path)
                    diarization_results = self.speaker_diarizer.get_summary(speaker_segments)
                    diarization_results['segments'] = speaker_segments
                    logger.info(f"Diarization: {diarization_results.get('speaker_count', 0)} speakers detected")
                except Exception as e:
                    logger.warning(f"Speaker diarization failed: {e}")

            # Step 7: Inappropriate language detection
            inappropriate_language = {}
            if self.language_detector:
                update_progress("Inappropriate language detection")
                try:
                    inappropriate_language = self.language_detector.analyze_with_whisper(
                        file_path, language=language
                    )
                    detected_count = inappropriate_language.get('detected_inappropriate_words', 0)
                    if detected_count > 0:
                        logger.warning(f"Detected {detected_count} inappropriate words")
                    else:
                        logger.info("No inappropriate language detected")
                except Exception as e:
                    logger.warning(f"Error in language detection: {e}")

            # Track which models were used
            models_used = []
            if hubert_used:
                models_used.append('hubert')
            if self.advanced_analyzer and self.advanced_analyzer.whisper_loaded:
                models_used.append('whisper')
            if self.speaker_diarizer:
                models_used.append('diarizer')

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

            logger.info(f"Analysis completed successfully! Duration: {audio_analysis['duration']:.1f}s")

            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            raise AudioAnalysisError(f"Analysis failed: {e}")
    
    def generate_report(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive report from analysis results.
        
        Args:
            analysis_results: Results from analyze_audio_file
            
        Returns:
            Generated report dictionary
            
        Raises:
            AudioAnalysisError: If report generation fails
        """
        logger.info("Generating comprehensive report")
        
        try:
            report = self.report_generator.generate_comprehensive_report(
                analysis_results, 
                analysis_results['file_path']
            )
            logger.info("Report generated successfully")
            return report
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise AudioAnalysisError(f"Report generation failed: {e}")
    
    def run_complete_analysis(
        self,
        file_path: str,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict:
        """Run complete analysis and generate report.
        
        Args:
            file_path: Path to audio file
            language: Language for analysis ('en' or 'he'). If None, uses instance language.
            progress_callback: Optional callback for progress updates
                             
        Returns:
            Complete analysis results with report
            
        Raises:
            AudioAnalysisError: If analysis or report generation fails
            
        Example:
            >>> analyzer = KindergartenRecordingAnalyzer(language='en')
            >>> results = analyzer.run_complete_analysis('recording.wav')
            >>> print(results['report']['summary']['overall_assessment'])
        """
        if language is None:
            language = self.language
            
        logger.info("=" * 60)
        logger.info("SoundShield-AI - Kindergarten Recording Analysis")
        logger.info("=" * 60)
        
        try:
            # Perform analysis
            analysis_results = self.analyze_audio_file(
                file_path,
                language=language,
                progress_callback=progress_callback
            )
            
            # Generate report
            report = self.generate_report(analysis_results)
            
            # Add report to results
            analysis_results['report'] = report
            
            # Print summary
            self._print_summary(analysis_results)
            
            return analysis_results
            
        except AudioAnalysisError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            raise AudioAnalysisError(f"Analysis error: {e}")
    
    def _print_summary(self, analysis_results: Dict) -> None:
        """Print analysis summary to console.
        
        Args:
            analysis_results: Complete analysis results with report
        """
        print("\n" + "=" * 60)
        print("Analysis Summary")
        print("=" * 60)
        
        report = analysis_results.get('report', {})
        summary = report.get('summary', {})
        
        print(f"Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
        print(f"Total Incidents: {summary.get('total_incidents', 0)}")
        print(f"Critical Incidents: {summary.get('critical_incidents', 0)}")
        print(f"Risk Level: {summary.get('risk_level', 'N/A')}")
        
        key_findings = summary.get('key_findings', [])
        if key_findings:
            print("\nKey Findings:")
            for finding in key_findings:
                print(f"  • {finding}")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for recommendation in recommendations:
                print(f"  • {recommendation}")
        
        print("\n" + "=" * 60)


def main() -> None:
    """Main function for command-line usage.
    
    Raises:
        SystemExit: On invalid arguments or analysis failure
    """
    if len(sys.argv) != 2:
        print("SoundShield-AI - Kindergarten Recording Analyzer")
        print("\nUsage: python main.py <path_to_audio_file>")
        print("\nExamples:")
        print("  python main.py recording.wav")
        print("  python main.py /path/to/kindergarten_recording.mp3")
        print(f"\nSupported formats: {', '.join(SUPPORTED_FORMATS)}")
        print(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Initialize analyzer
        logger.info("Initializing SoundShield-AI Analyzer")
        analyzer = KindergartenRecordingAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(file_path)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"Results saved in 'reports' directory")
        
        logger.info("Analysis session completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except InvalidAudioFormatError as e:
        logger.error(f"Invalid audio format: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except AudioAnalysisError as e:
        logger.error(f"Analysis error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
