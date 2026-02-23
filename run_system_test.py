"""
SoundShield-AI System Test

Runs the full analysis pipeline on synthetic test audio to verify
all modules work correctly end-to-end.
"""

import sys
import os
import time
import traceback

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def test_audio_analyzer(file_path):
    """Test the basic audio analyzer."""
    print("\n" + "=" * 60)
    print("TEST 1: Audio Analyzer")
    print("=" * 60)
    
    from audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    
    result = analyzer.analyze_audio_file(file_path)
    
    print(f"  Duration: {result.get('duration', 'N/A'):.2f} seconds")
    print(f"  Sample Rate: {result.get('sample_rate', 'N/A')} Hz")
    print(f"  Segments: {len(result.get('segments', []))}")
    
    # Also test load_audio
    audio, sr = analyzer.load_audio(file_path)
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio min/max: {audio.min():.4f} / {audio.max():.4f}")
    
    print("  [PASS] Audio Analyzer works!")
    return result, audio, sr


def test_emotion_detector(segments, sr):
    """Test the emotion detector."""
    print("\n" + "=" * 60)
    print("TEST 2: Emotion Detector")
    print("=" * 60)
    
    from emotion_detector import EmotionDetector
    detector = EmotionDetector()
    
    # Analyze segments
    emotion_results = detector.analyze_segment_emotions(segments, sr)
    print(f"  Analyzed {len(emotion_results)} segments")
    
    for result in emotion_results:
        analysis = result['emotion_analysis']
        print(f"    Segment {result['segment_index']}: "
              f"{analysis['primary_emotion']} "
              f"(confidence: {analysis['confidence']:.2f})")
        scores = analysis['emotion_scores']
        for emotion, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = '#' * int(score * 20)
            print(f"      {emotion:12s}: {score:.3f} {bar}")
    
    # Test concerning emotions
    concerning = detector.detect_concerning_emotions(emotion_results)
    print(f"\n  Concerning segments: {len(concerning)}")
    for c in concerning:
        print(f"    [{c['start_time']:.1f}s-{c['end_time']:.1f}s] "
              f"{c['detected_emotion']} "
              f"(severity: {c['severity']}, confidence: {c['confidence']:.2f})")
    
    print("  [PASS] Emotion Detector works!")
    return emotion_results, concerning


def test_cry_detector(audio, sr):
    """Test the cry detector."""
    print("\n" + "=" * 60)
    print("TEST 3: Cry Detector")
    print("=" * 60)
    
    from cry_detector import CryDetector
    detector = CryDetector()
    
    # Detect cry segments
    cry_segments = detector.detect_cry_segments(audio, sr)
    print(f"  Detected cry segments: {len(cry_segments)}")
    
    for cry in cry_segments:
        print(f"    [{cry['start_time']:.1f}s-{cry['end_time']:.1f}s] "
              f"intensity: {cry['intensity']}, "
              f"confidence: {cry['confidence']:.2f}")
    
    # Check for responses
    cry_with_responses = detector.detect_response_to_cry(audio, sr, cry_segments)
    print(f"\n  Cry response analysis: {len(cry_with_responses)} segments")
    for cry in cry_with_responses:
        response = "YES" if cry.get('response_detected', False) else "NO"
        quality = cry.get('response_quality', 'N/A')
        print(f"    [{cry['start_time']:.1f}s-{cry['end_time']:.1f}s] "
              f"Response: {response}, Quality: {quality}")
    
    print("  [PASS] Cry Detector works!")
    return cry_segments, cry_with_responses


def test_violence_detector(audio, sr):
    """Test the violence detector."""
    print("\n" + "=" * 60)
    print("TEST 4: Violence Detector")
    print("=" * 60)
    
    from violence_detector import ViolenceDetector
    detector = ViolenceDetector()
    
    violence_segments = detector.detect_violence_segments(audio, sr)
    print(f"  Detected violence segments: {len(violence_segments)}")
    
    for seg in violence_segments:
        print(f"    [{seg.get('start_time', 0):.1f}s-{seg.get('end_time', 0):.1f}s] "
              f"type: {seg.get('type', 'N/A')}, "
              f"confidence: {seg.get('confidence', 0):.2f}")
    
    print("  [PASS] Violence Detector works!")
    return violence_segments


def test_neglect_detector(audio, sr, cry_segments, violence_segments):
    """Test the neglect detector."""
    print("\n" + "=" * 60)
    print("TEST 5: Neglect Detector")
    print("=" * 60)
    
    from neglect_detector import NeglectDetector
    detector = NeglectDetector()
    
    neglect = detector.detect_neglect_patterns(audio, sr, cry_segments, violence_segments)
    print(f"  Neglect analysis keys: {list(neglect.keys()) if isinstance(neglect, dict) else type(neglect)}")
    
    if isinstance(neglect, dict):
        for key, value in neglect.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"    {key}: {value}")
            elif isinstance(value, list):
                print(f"    {key}: {len(value)} items")
            elif isinstance(value, dict):
                print(f"    {key}: {len(value)} keys")
    
    print("  [PASS] Neglect Detector works!")
    return neglect


def test_report_generator(analysis_results, file_path):
    """Test the report generator."""
    print("\n" + "=" * 60)
    print("TEST 6: Report Generator")
    print("=" * 60)
    
    from report_generator import ReportGenerator
    generator = ReportGenerator()
    
    report = generator.generate_comprehensive_report(analysis_results, file_path)
    print(f"  Report keys: {list(report.keys()) if isinstance(report, dict) else type(report)}")
    
    if isinstance(report, dict):
        summary = report.get('summary', {})
        print(f"\n  Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
        print(f"  Total Incidents: {summary.get('total_incidents', 0)}")
        print(f"  Critical Incidents: {summary.get('critical_incidents', 0)}")
        print(f"  Risk Level: {summary.get('risk_level', 'N/A')}")
        
        findings = summary.get('key_findings', [])
        if findings:
            print(f"\n  Key Findings:")
            for f in findings[:5]:
                print(f"    - {f}")
        
        recs = report.get('recommendations', [])
        if recs:
            print(f"\n  Recommendations:")
            for r in recs[:5]:
                print(f"    - {r}")
    
    print("  [PASS] Report Generator works!")
    return report


def run_full_pipeline(file_path):
    """Run the complete analysis pipeline."""
    print("\n" + "#" * 60)
    print(f"  FULL PIPELINE TEST: {os.path.basename(file_path)}")
    print("#" * 60)
    
    start_time = time.time()
    
    # Test 1: Audio Analysis
    audio_result, audio, sr = test_audio_analyzer(file_path)
    
    # Test 2: Emotion Detection
    emotion_results, concerning = test_emotion_detector(
        audio_result.get('segments', []), sr
    )
    
    # Test 3: Cry Detection
    cry_segments, cry_responses = test_cry_detector(audio, sr)
    
    # Test 4: Violence Detection
    violence_segments = test_violence_detector(audio, sr)
    
    # Test 5: Neglect Detection
    neglect = test_neglect_detector(audio, sr, cry_segments, violence_segments)
    
    # Compile results for report
    analysis_results = {
        'file_path': file_path,
        'duration': audio_result.get('duration', 0),
        'audio_analysis': audio_result,
        'emotion_results': emotion_results,
        'concerning_emotions': concerning,
        'cry_segments': cry_segments,
        'cry_with_responses': cry_responses,
        'violence_segments': violence_segments,
        'neglect_analysis': neglect,
        'advanced_analysis': {},
        'inappropriate_language': {},
        'analysis_timestamp': time.time(),
        'language': 'en'
    }
    
    # Test 6: Report Generation
    report = test_report_generator(analysis_results, file_path)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"  ALL TESTS PASSED for {os.path.basename(file_path)}!")
    print(f"  Total time: {elapsed:.2f} seconds")
    print("=" * 60)
    
    return analysis_results, report


def main():
    """Run all tests."""
    print("=" * 60)
    print("  SOUNDSHIELD-AI - FULL SYSTEM TEST")
    print("=" * 60)
    
    test_files = [
        'tests/fixtures/test_sample.wav',
        'tests/fixtures/test_aggressive.wav',
        'tests/fixtures/test_short.wav',
    ]
    
    results = {}
    failures = []
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"\n[SKIP] {file_path} not found")
            continue
        
        try:
            analysis, report = run_full_pipeline(file_path)
            results[file_path] = (analysis, report)
        except Exception as e:
            print(f"\n[FAIL] {file_path}: {e}")
            traceback.print_exc()
            failures.append((file_path, str(e)))
    
    # Final summary
    print("\n" + "#" * 60)
    print("  FINAL SUMMARY")
    print("#" * 60)
    print(f"  Files tested: {len(results) + len(failures)}")
    print(f"  Passed: {len(results)}")
    print(f"  Failed: {len(failures)}")
    
    if failures:
        print("\n  FAILURES:")
        for path, error in failures:
            print(f"    - {path}: {error}")
    
    if results:
        print("\n  RESULTS:")
        for path, (analysis, report) in results.items():
            name = os.path.basename(path)
            summary = report.get('summary', {}) if isinstance(report, dict) else {}
            risk = summary.get('risk_level', 'N/A')
            incidents = summary.get('total_incidents', 0)
            assessment = summary.get('overall_assessment', 'N/A')
            print(f"    {name}:")
            print(f"      Assessment: {assessment}")
            print(f"      Risk Level: {risk}")
            print(f"      Incidents: {incidents}")
    
    print("\n" + "#" * 60)
    if not failures:
        print("  ALL SYSTEMS OPERATIONAL!")
    else:
        print("  SOME TESTS FAILED - SEE ABOVE")
    print("#" * 60)
    
    return len(failures) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
