"""
Test and Compare Advanced Models
בדיקה והשוואת מודלים מתקדמים
"""

import os
import json
from datetime import datetime
from advanced_models_research import AdvancedModelsResearch
from advanced_analyzer import AdvancedAnalyzer

def main():
    """
    Test all advanced models and generate comprehensive report
    בדיקת כל המודלים המתקדמים ויצירת דוח מקיף
    """
    print("=" * 80)
    print("בדיקה והשוואת מודלים מתקדמים למערכת ניתוח הקלטות גן ילדים")
    print("Advanced Models Testing and Comparison for Kindergarten Recording Analyzer")
    print("=" * 80)
    print()
    
    # Find test audio file
    test_file = "sample_kindergarten_recording.wav"
    if not os.path.exists(test_file):
        # Try to find any audio file
        for ext in ['.wav', '.mp3', '.m4a']:
            for file in os.listdir('.'):
                if file.endswith(ext):
                    test_file = file
                    break
            if test_file and os.path.exists(test_file):
                break
        
        if not os.path.exists(test_file):
            print("❌ לא נמצא קובץ אודיו לבדיקה")
            print("❌ No audio file found for testing")
            print("   צור קובץ דוגמה עם: py example_usage.py")
            print("   Create sample file with: py example_usage.py")
            return
    
    print(f"📁 קובץ נבדק: {test_file}")
    print(f"📁 Test file: {test_file}")
    print()
    
    # Initialize research module
    researcher = AdvancedModelsResearch()
    
    # Run evaluation
    print("🔬 מתחיל בדיקת מודלים...")
    print("🔬 Starting model evaluation...")
    print()
    
    results = researcher.evaluate_models(test_file)
    
    # Generate report
    print()
    print("=" * 80)
    report_text = researcher.generate_comparison_report(results)
    print(report_text)
    print("=" * 80)
    
    # Save results to file
    report_file = f"models_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 דוח נשמר: {report_file}")
    print(f"💾 Report saved: {report_file}")
    
    # Test advanced analyzer
    print("\n" + "=" * 80)
    print("מבצע ניתוח מתקדם...")
    print("Running Advanced Analysis...")
    print("=" * 80)
    
    advanced_analyzer = AdvancedAnalyzer(use_whisper=True, use_transformer_emotion=True)
    advanced_analyzer.load_models()
    
    if advanced_analyzer.models_loaded:
        advanced_results = advanced_analyzer.comprehensive_analysis(test_file)
        
        print("\n📊 תוצאות ניתוח מתקדם:")
        print("📊 Advanced Analysis Results:")
        print("-" * 80)
        
        if advanced_results.get('whisper_analysis'):
            whisper = advanced_results['whisper_analysis']
            print(f"\n📝 Whisper Transcription:")
            if 'transcript' in whisper:
                print(f"   אורך תמלול: {whisper.get('transcript_length', 0)} תווים")
                print(f"   Transcript length: {whisper.get('transcript_length', 0)} characters")
            if 'concerning_segments' in whisper:
                print(f"   קטעים מדאיגים: {len(whisper['concerning_segments'])}")
                print(f"   Concerning segments: {len(whisper['concerning_segments'])}")
                for seg in whisper['concerning_segments'][:3]:
                    print(f"      - {seg['start_time']:.1f}s-{seg['end_time']:.1f}s: {seg['text'][:50]}...")
        
        if advanced_results.get('emotion_analysis'):
            emotions = advanced_results['emotion_analysis']
            print(f"\n😊 Emotion Analysis:")
            if 'top_emotions' in emotions:
                print(f"   רגשות מובילים:")
                print(f"   Top emotions:")
                for emo, score in list(emotions['top_emotions'].items())[:5]:
                    print(f"      - {emo}: {score:.3f}")
            if 'concerning_emotions' in emotions and emotions['concerning_emotions']:
                print(f"   ⚠️ רגשות מדאיגים:")
                print(f"   ⚠️ Concerning emotions:")
                for emo, score in emotions['concerning_emotions'].items():
                    print(f"      - {emo}: {score:.3f}")
        
        if advanced_results.get('combined_insights'):
            insights = advanced_results['combined_insights']
            print(f"\n🎯 Combined Insights:")
            print(f"   ציון סיכון: {insights.get('risk_score', 0):.2f}")
            print(f"   Risk score: {insights.get('risk_score', 0):.2f}")
            print(f"   ביטחון: {insights.get('confidence', 0):.2f}")
            print(f"   Confidence: {insights.get('confidence', 0):.2f}")
            if insights.get('concerns'):
                print(f"   חששות זוהו: {len(insights['concerns'])}")
                print(f"   Concerns detected: {len(insights['concerns'])}")
        
        # Save advanced results
        advanced_file = f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(advanced_file, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 תוצאות מתקדמות נשמרו: {advanced_file}")
        print(f"💾 Advanced results saved: {advanced_file}")
    
    # Final recommendations
    print("\n" + "=" * 80)
    print("✅ המלצות סופיות / Final Recommendations:")
    print("=" * 80)
    
    successful = [name for name, result in results['performance_comparison'].items() 
                 if result.get('status') == 'success']
    
    if successful:
        print("\n✅ מודלים מומלצים לשימוש:")
        print("✅ Recommended models for use:")
        for model in successful:
            print(f"   • {model}")
            result = results['performance_comparison'][model]
            print(f"     דיוק משוער: {result.get('accuracy_estimate', 0)*100:.1f}%")
            print(f"     Estimated accuracy: {result.get('accuracy_estimate', 0)*100:.1f}%")
    else:
        print("\n⚠️ לא נמצאו מודלים פעילים")
        print("⚠️ No active models found")
        print("\n💡 התקן מודלים:")
        print("💡 Install models:")
        print("   pip install openai-whisper")
        print("   pip install transformers torch")
        print("   pip install pyannote.audio")
    
    print("\n" + "=" * 80)
    print("✅ בדיקה הושלמה!")
    print("✅ Testing completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
