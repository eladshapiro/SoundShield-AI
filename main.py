"""
Main Application for Kindergarten Recording Analyzer
אפליקציה ראשית למערכת ניתוח הקלטות גן ילדים
"""

import os
import sys
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from audio_analyzer import AudioAnalyzer
from emotion_detector import EmotionDetector
from cry_detector import CryDetector
from violence_detector import ViolenceDetector
from neglect_detector import NeglectDetector
from report_generator import ReportGenerator

class KindergartenRecordingAnalyzer:
    def __init__(self):
        """
        Initialize the main analyzer application
        אתחול אפליקציית הניתוח הראשית
        """
        print("מאתחל מערכת ניתוח הקלטות גן ילדים...")
        print("Initializing Kindergarten Recording Analyzer...")
        
        # Initialize all analysis modules
        self.audio_analyzer = AudioAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.cry_detector = CryDetector()
        self.violence_detector = ViolenceDetector()
        self.neglect_detector = NeglectDetector()
        self.report_generator = ReportGenerator()
        
        print("כל המודולים אותחלו בהצלחה!")
        print("All modules initialized successfully!")
    
    def analyze_audio_file(self, file_path: str) -> Dict:
        """
        Perform complete analysis of an audio file
        ביצוע ניתוח מלא של קובץ אודיו
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\nמתחיל ניתוח קובץ: {file_path}")
        print(f"Starting analysis of file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"קובץ לא נמצא: {file_path}")
        
        # Step 1: Basic audio analysis
        print("\nשלב 1: ניתוח אודיו בסיסי")
        print("Step 1: Basic audio analysis")
        audio_analysis = self.audio_analyzer.analyze_audio_file(file_path)
        
        # Step 2: Emotion detection
        print("\nשלב 2: זיהוי רגשות")
        print("Step 2: Emotion detection")
        emotion_results = self.emotion_detector.analyze_segment_emotions(
            audio_analysis['segments'], 
            audio_analysis['sample_rate']
        )
        concerning_emotions = self.emotion_detector.detect_concerning_emotions(emotion_results)
        
        # Step 3: Cry detection
        print("\nשלב 3: זיהוי בכי תינוקות")
        print("Step 3: Baby cry detection")
        audio, sr = self.audio_analyzer.load_audio(file_path)
        cry_segments = self.cry_detector.detect_cry_segments(audio, sr)
        cry_with_responses = self.cry_detector.detect_response_to_cry(audio, sr, cry_segments)
        
        # Step 4: Violence detection
        print("\nשלב 4: זיהוי אלימות")
        print("Step 4: Violence detection")
        violence_segments = self.violence_detector.detect_violence_segments(audio, sr)
        
        # Step 5: Neglect detection
        print("\nשלב 5: זיהוי הזנחה")
        print("Step 5: Neglect detection")
        neglect_analysis = self.neglect_detector.detect_neglect_patterns(
            audio, sr, cry_segments, violence_segments
        )
        
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
            'analysis_timestamp': time.time()
        }
        
        print(f"\nניתוח הושלם בהצלחה!")
        print(f"Analysis completed successfully!")
        print(f"משך הקלטה: {audio_analysis['duration']:.1f} שניות")
        print(f"Recording duration: {audio_analysis['duration']:.1f} seconds")
        
        return analysis_results
    
    def generate_report(self, analysis_results: Dict) -> Dict:
        """
        Generate comprehensive report from analysis results
        יצירת דוח מקיף מתוצאות הניתוח
        
        Args:
            analysis_results: Results from analyze_audio_file
            
        Returns:
            Generated report dictionary
        """
        print("\nיוצר דוח מקיף...")
        print("Generating comprehensive report...")
        
        report = self.report_generator.generate_comprehensive_report(
            analysis_results, 
            analysis_results['file_path']
        )
        
        print("דוח נוצר בהצלחה!")
        print("Report generated successfully!")
        
        return report
    
    def run_complete_analysis(self, file_path: str) -> Dict:
        """
        Run complete analysis and generate report
        הרצת ניתוח מלא ויצירת דוח
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Complete analysis results with report
        """
        print("=" * 60)
        print("מערכת ניתוח הקלטות גן ילדים")
        print("Kindergarten Recording Analyzer")
        print("=" * 60)
        
        try:
            # Perform analysis
            analysis_results = self.analyze_audio_file(file_path)
            
            # Generate report
            report = self.generate_report(analysis_results)
            
            # Add report to results
            analysis_results['report'] = report
            
            # Print summary
            self._print_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            print(f"\nשגיאה בניתוח: {str(e)}")
            print(f"Analysis error: {str(e)}")
            raise
    
    def _print_summary(self, analysis_results: Dict):
        """
        Print analysis summary to console
        הדפסת סיכום הניתוח לקונסול
        """
        print("\n" + "=" * 60)
        print("סיכום ניתוח / Analysis Summary")
        print("=" * 60)
        
        report = analysis_results.get('report', {})
        summary = report.get('summary', {})
        
        print(f"הערכה כללית: {summary.get('overall_assessment', 'לא זמין')}")
        print(f"Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
        
        print(f"סה\"כ אירועים: {summary.get('total_incidents', 0)}")
        print(f"Total Incidents: {summary.get('total_incidents', 0)}")
        
        print(f"אירועים קריטיים: {summary.get('critical_incidents', 0)}")
        print(f"Critical Incidents: {summary.get('critical_incidents', 0)}")
        
        print(f"רמת סיכון: {summary.get('risk_level', 'לא זמין')}")
        print(f"Risk Level: {summary.get('risk_level', 'N/A')}")
        
        print("\nממצאים עיקריים / Key Findings:")
        for finding in summary.get('key_findings', []):
            print(f"  • {finding}")
        
        print("\nהמלצות / Recommendations:")
        for recommendation in report.get('recommendations', []):
            print(f"  • {recommendation}")
        
        print("\n" + "=" * 60)

def main():
    """
    Main function for command-line usage
    פונקציה ראשית לשימוש בשורת הפקודה
    """
    if len(sys.argv) != 2:
        print("שימוש: python main.py <path_to_audio_file>")
        print("Usage: python main.py <path_to_audio_file>")
        print("\nדוגמאות / Examples:")
        print("  python main.py recording.wav")
        print("  python main.py /path/to/kindergarten_recording.mp3")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = KindergartenRecordingAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(file_path)
        
        print(f"\nניתוח הושלם בהצלחה! תוצאות נשמרו בתיקיית 'reports'")
        print(f"Analysis completed successfully! Results saved in 'reports' directory")
        
    except Exception as e:
        print(f"\nשגיאה: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
