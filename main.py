"""
Main Application for Kindergarten Recording Analyzer
"""

import os
import sys
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import our custom modules
from audio_analyzer import AudioAnalyzer
from emotion_detector import EmotionDetector
from cry_detector import CryDetector
from violence_detector import ViolenceDetector
from neglect_detector import NeglectDetector
from report_generator import ReportGenerator

# Import advanced models (optional)
try:
    from advanced_analyzer import AdvancedAnalyzer
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("⚠️ Advanced models not available (optional)")

# Import inappropriate language detector
try:
    from inappropriate_language_detector import InappropriateLanguageDetector
    LANGUAGE_DETECTOR_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTOR_AVAILABLE = False

class KindergartenRecordingAnalyzer:
    def __init__(self, language: str = 'en'):
        """
        Initialize the main analyzer application
        
        Args:
            language: Language for analysis ('en' for English, 'he' for Hebrew)
        """
        self.language = language
        print("Initializing Kindergarten Recording Analyzer...")
        
        # Initialize all analysis modules
        self.audio_analyzer = AudioAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.cry_detector = CryDetector()
        self.violence_detector = ViolenceDetector()
        self.neglect_detector = NeglectDetector()
        self.report_generator = ReportGenerator()
        
        # Initialize advanced analyzer if available
        self.advanced_analyzer = None
        if ADVANCED_MODELS_AVAILABLE:
            try:
                self.advanced_analyzer = AdvancedAnalyzer(use_whisper=True, use_transformer_emotion=True)
                self.advanced_analyzer.load_models()
                if self.advanced_analyzer.models_loaded:
                    print("✅ Advanced models loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading advanced models: {e}")
        
        # Initialize inappropriate language detector
        self.language_detector = None
        if LANGUAGE_DETECTOR_AVAILABLE:
            try:
                self.language_detector = InappropriateLanguageDetector()
                print("✅ Inappropriate language detector initialized")
            except Exception as e:
                print(f"⚠️ Error initializing language detector: {e}")
        
        print("All modules initialized successfully!")
    
    def analyze_audio_file(self, file_path: str, language: Optional[str] = None) -> Dict:
        """
        Perform complete analysis of an audio file
        
        Args:
            file_path: Path to audio file
            language: Language for analysis ('en' or 'he'). If None, uses instance language.
            
        Returns:
            Dictionary containing all analysis results
        """
        if language is None:
            language = self.language
            
        print(f"\nStarting analysis of file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Step 1: Basic audio analysis
        print("\nStep 1: Basic audio analysis")
        audio_analysis = self.audio_analyzer.analyze_audio_file(file_path)
        
        # Step 2: Emotion detection
        print("\nStep 2: Emotion detection")
        emotion_results = self.emotion_detector.analyze_segment_emotions(
            audio_analysis['segments'], 
            audio_analysis['sample_rate']
        )
        concerning_emotions = self.emotion_detector.detect_concerning_emotions(emotion_results)
        
        # Step 3: Cry detection
        print("\nStep 3: Baby cry detection")
        audio, sr = self.audio_analyzer.load_audio(file_path)
        cry_segments = self.cry_detector.detect_cry_segments(audio, sr)
        cry_with_responses = self.cry_detector.detect_response_to_cry(audio, sr, cry_segments)
        
        # Step 4: Violence detection
        print("\nStep 4: Violence detection")
        violence_segments = self.violence_detector.detect_violence_segments(audio, sr)
        
        # Step 5: Neglect detection
        print("\nStep 5: Neglect detection")
        neglect_analysis = self.neglect_detector.detect_neglect_patterns(
            audio, sr, cry_segments, violence_segments
        )
        
        # Advanced analysis with ML models (if available)
        advanced_analysis = {}
        if self.advanced_analyzer and self.advanced_analyzer.models_loaded:
            print("\nStep 6: Advanced analysis with ML models")
            try:
                advanced_analysis = self.advanced_analyzer.comprehensive_analysis(file_path, language=language)
                print("✅ Advanced analysis completed")
            except Exception as e:
                print(f"⚠️ Error in advanced analysis: {e}")
        
        # Inappropriate language detection
        inappropriate_language = {}
        if self.language_detector:
            print("\nStep 7: Inappropriate language and profanity detection")
            try:
                inappropriate_language = self.language_detector.analyze_with_whisper(file_path, language=language)
                if inappropriate_language.get('detected_inappropriate_words', 0) > 0:
                    print(f"⚠️ Detected {inappropriate_language['detected_inappropriate_words']} inappropriate words")
                else:
                    print("✅ No inappropriate language detected")
            except Exception as e:
                print(f"⚠️ Error in language detection: {e}")
        
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
        
        print(f"\nAnalysis completed successfully!")
        print(f"Recording duration: {audio_analysis['duration']:.1f} seconds")
        
        return analysis_results
    
    def generate_report(self, analysis_results: Dict) -> Dict:
        """
        Generate comprehensive report from analysis results
        
        Args:
            analysis_results: Results from analyze_audio_file
            
        Returns:
            Generated report dictionary
        """
        print("\nGenerating comprehensive report...")
        
        report = self.report_generator.generate_comprehensive_report(
            analysis_results, 
            analysis_results['file_path']
        )
        
        print("Report generated successfully!")
        
        return report
    
    def run_complete_analysis(self, file_path: str, language: Optional[str] = None) -> Dict:
        """
        Run complete analysis and generate report
        
        Args:
            file_path: Path to audio file
            language: Language for analysis ('en' or 'he'). If None, uses instance language.
            
        Returns:
            Complete analysis results with report
        """
        if language is None:
            language = self.language
            
        print("=" * 60)
        print("Kindergarten Recording Analyzer")
        print("=" * 60)
        
        try:
            # Perform analysis
            analysis_results = self.analyze_audio_file(file_path, language=language)
            
            # Generate report
            report = self.generate_report(analysis_results)
            
            # Add report to results
            analysis_results['report'] = report
            
            # Print summary
            self._print_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            print(f"\nAnalysis error: {str(e)}")
            raise
    
    def _print_summary(self, analysis_results: Dict):
        """
        Print analysis summary to console
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
        
        print("\nKey Findings:")
        for finding in summary.get('key_findings', []):
            print(f"  • {finding}")
        
        print("\nRecommendations:")
        for recommendation in report.get('recommendations', []):
            print(f"  • {recommendation}")
        
        print("\n" + "=" * 60)

def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_audio_file>")
        print("\nExamples:")
        print("  python main.py recording.wav")
        print("  python main.py /path/to/kindergarten_recording.mp3")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = KindergartenRecordingAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(file_path)
        
        print(f"\nAnalysis completed successfully! Results saved in 'reports' directory")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
