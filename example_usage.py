"""
Example Usage of Kindergarten Recording Analyzer
דוגמת שימוש במערכת ניתוח הקלטות גן ילדים
"""

import os
import numpy as np
import librosa
from main import KindergartenRecordingAnalyzer

def create_sample_audio():
    """
    Create a sample audio file for testing
    יצירת קובץ אודיו לדוגמה לבדיקה
    """
    print("יוצר קובץ אודיו לדוגמה...")
    print("Creating sample audio file...")
    
    # Create a 30-second sample with different audio patterns
    duration = 30  # seconds
    sr = 22050  # sample rate
    
    # Generate different segments
    segments = []
    
    # Segment 1: Normal background noise (0-5 seconds)
    normal_noise = np.random.normal(0, 0.01, sr * 5)
    segments.append(normal_noise)
    
    # Segment 2: Baby crying simulation (5-10 seconds)
    # High frequency, variable amplitude
    t = np.linspace(0, 5, sr * 5)
    cry_freq = 400 + 200 * np.sin(2 * np.pi * 0.5 * t)  # Variable frequency
    baby_cry = 0.3 * np.sin(2 * np.pi * cry_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    segments.append(baby_cry)
    
    # Segment 3: Adult speech simulation (10-15 seconds)
    # Lower frequency, speech-like characteristics
    t = np.linspace(0, 5, sr * 5)
    speech_freq = 150 + 50 * np.sin(2 * np.pi * 0.3 * t)
    adult_speech = 0.2 * np.sin(2 * np.pi * speech_freq * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t))
    segments.append(adult_speech)
    
    # Segment 4: Potential shouting/aggression (15-18 seconds)
    # High energy, high frequency
    t = np.linspace(0, 3, sr * 3)
    shout_freq = 600 + 300 * np.sin(2 * np.pi * 0.8 * t)
    shouting = 0.4 * np.sin(2 * np.pi * shout_freq * t)
    segments.append(shouting)
    
    # Segment 5: Silence (18-25 seconds)
    silence = np.zeros(sr * 7)
    segments.append(silence)
    
    # Segment 6: Continued crying (25-30 seconds)
    t = np.linspace(0, 5, sr * 5)
    cry_freq = 450 + 150 * np.sin(2 * np.pi * 0.6 * t)
    continued_cry = 0.25 * np.sin(2 * np.pi * cry_freq * t) * (0.4 + 0.6 * np.sin(2 * np.pi * 1.5 * t))
    segments.append(continued_cry)
    
    # Combine all segments
    audio = np.concatenate(segments)
    
    # Add some noise
    noise = np.random.normal(0, 0.005, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save as WAV file
    sample_file = "sample_kindergarten_recording.wav"
    import soundfile as sf
    sf.write(sample_file, audio, sr)
    
    print(f"קובץ דוגמה נוצר: {sample_file}")
    print(f"Sample file created: {sample_file}")
    
    return sample_file

def run_example_analysis():
    """
    Run example analysis on sample audio
    הרצת ניתוח דוגמה על אודיו לדוגמה
    """
    print("=" * 60)
    print("דוגמת שימוש במערכת ניתוח הקלטות גן ילדים")
    print("Example Usage of Kindergarten Recording Analyzer")
    print("=" * 60)
    
    try:
        # Create sample audio if it doesn't exist
        sample_file = "sample_kindergarten_recording.wav"
        if not os.path.exists(sample_file):
            sample_file = create_sample_audio()
        
        # Initialize analyzer
        print("\nמאתחל מנתח...")
        print("Initializing analyzer...")
        analyzer = KindergartenRecordingAnalyzer()
        
        # Run analysis
        print(f"\nמתחיל ניתוח קובץ: {sample_file}")
        print(f"Starting analysis of file: {sample_file}")
        results = analyzer.run_complete_analysis(sample_file)
        
        # Print detailed results
        print_detailed_results(results)
        
        return results
        
    except Exception as e:
        print(f"\nשגיאה בדוגמה: {str(e)}")
        print(f"Error in example: {str(e)}")
        raise

def print_detailed_results(results):
    """
    Print detailed analysis results
    הדפסת תוצאות ניתוח מפורטות
    """
    print("\n" + "=" * 60)
    print("תוצאות מפורטות / Detailed Results")
    print("=" * 60)
    
    # Audio analysis
    audio_analysis = results['audio_analysis']
    print(f"\nניתוח אודיו בסיסי:")
    print(f"Basic Audio Analysis:")
    print(f"  - משך הקלטה: {audio_analysis['duration']:.1f} שניות")
    print(f"  - Recording Duration: {audio_analysis['duration']:.1f} seconds")
    print(f"  - קטעים שקטים: {len(audio_analysis['silent_segments'])}")
    print(f"  - Silent Segments: {len(audio_analysis['silent_segments'])}")
    print(f"  - קטעים רועשים: {len(audio_analysis['loud_segments'])}")
    print(f"  - Loud Segments: {len(audio_analysis['loud_segments'])}")
    
    # Emotion analysis
    concerning_emotions = results['concerning_emotions']
    print(f"\nניתוח רגשי:")
    print(f"Emotion Analysis:")
    print(f"  - רגשות מדאיגים: {len(concerning_emotions)}")
    print(f"  - Concerning Emotions: {len(concerning_emotions)}")
    
    for i, emotion in enumerate(concerning_emotions, 1):
        print(f"    {i}. זמן: {emotion['start_time']:.1f}-{emotion['end_time']:.1f}s")
        print(f"       רגש: {emotion['detected_emotion']}, חומרה: {emotion['severity']}")
        print(f"       Emotion: {emotion['detected_emotion']}, Severity: {emotion['severity']}")
    
    # Cry analysis
    cry_segments = results['cry_segments']
    cry_with_responses = results['cry_with_responses']
    print(f"\nניתוח בכי:")
    print(f"Cry Analysis:")
    print(f"  - קטעי בכי זוהו: {len(cry_segments)}")
    print(f"  - Cry Segments Detected: {len(cry_segments)}")
    print(f"  - תגובות זוהו: {sum(1 for cry in cry_with_responses if cry['response_detected'])}")
    print(f"  - Responses Detected: {sum(1 for cry in cry_with_responses if cry['response_detected'])}")
    
    for i, cry in enumerate(cry_with_responses, 1):
        response_status = "עם תגובה" if cry['response_detected'] else "ללא תגובה"
        print(f"    {i}. זמן: {cry['start_time']:.1f}-{cry['end_time']:.1f}s")
        print(f"       עוצמה: {cry['intensity']}, {response_status}")
        print(f"       Intensity: {cry['intensity']}, {response_status}")
    
    # Violence analysis
    violence_segments = results['violence_segments']
    print(f"\nניתוח אלימות:")
    print(f"Violence Analysis:")
    print(f"  - אירועי אלימות: {len(violence_segments)}")
    print(f"  - Violence Incidents: {len(violence_segments)}")
    
    for i, violence in enumerate(violence_segments, 1):
        violence_types = ', '.join(violence['violence_types'])
        print(f"    {i}. זמן: {violence['start_time']:.1f}-{violence['end_time']:.1f}s")
        print(f"       סוגים: {violence_types}")
        print(f"       חומרה: {violence['adjusted_severity']}")
        print(f"       Types: {violence_types}")
        print(f"       Severity: {violence['adjusted_severity']}")
    
    # Neglect analysis
    neglect_analysis = results['neglect_analysis']
    print(f"\nניתוח הזנחה:")
    print(f"Neglect Analysis:")
    print(f"  - רמת הזנחה: {neglect_analysis['neglect_severity']}")
    print(f"  - Neglect Level: {neglect_analysis['neglect_severity']}")
    print(f"  - ציון הזנחה: {neglect_analysis['overall_neglect_score']:.2f}")
    print(f"  - Neglect Score: {neglect_analysis['overall_neglect_score']:.2f}")
    print(f"  - בכי ללא תגובה: {len(neglect_analysis['unanswered_cries'])}")
    print(f"  - Unanswered Cries: {len(neglect_analysis['unanswered_cries'])}")
    
    # Report summary
    report = results['report']
    summary = report['summary']
    print(f"\nסיכום דוח:")
    print(f"Report Summary:")
    print(f"  - הערכה כללית: {summary['overall_assessment']}")
    print(f"  - Overall Assessment: {summary['overall_assessment']}")
    print(f"  - רמת סיכון: {summary['risk_level']}")
    print(f"  - Risk Level: {summary['risk_level']}")
    print(f"  - סה\"כ אירועים: {summary['total_incidents']}")
    print(f"  - Total Incidents: {summary['total_incidents']}")
    
    print(f"\nהמלצות:")
    print(f"Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")

if __name__ == "__main__":
    # Run the example
    try:
        results = run_example_analysis()
        print(f"\nדוגמה הושלמה בהצלחה!")
        print(f"Example completed successfully!")
        print(f"תוצאות נשמרו בתיקיית 'reports'")
        print(f"Results saved in 'reports' directory")
        
    except Exception as e:
        print(f"\nשגיאה בדוגמה: {str(e)}")
        print(f"Example error: {str(e)}")
        import traceback
        traceback.print_exc()
