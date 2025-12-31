"""
Advanced Analyzer with State-of-the-Art Models
"""

import numpy as np
import librosa
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyzer:
    """
    Advanced analyzer using best-in-class models
    """
    
    def __init__(self, use_whisper: bool = True, use_transformer_emotion: bool = True):
        """
        Initialize advanced analyzer
        
        Args:
            use_whisper: Use Whisper for transcription
            use_transformer_emotion: Use Transformers for emotion detection
        """
        self.use_whisper = use_whisper
        self.use_transformer_emotion = use_transformer_emotion
        self.whisper_model = None
        self.emotion_model = None
        self.models_loaded = False
        
        print("Initializing Advanced Analyzer...")
        
    def load_models(self):
        """Load advanced models if available"""
        try:
            if self.use_whisper:
                try:
                    import whisper
                    print("  Loading Whisper...")
                    self.whisper_model = whisper.load_model("base")
                    print("  ✅ Whisper loaded successfully")
                except ImportError:
                    print("  ⚠️ Whisper not installed - skipping")
                    self.use_whisper = False
                except Exception as e:
                    print(f"  ⚠️ Error loading Whisper: {e}")
                    self.use_whisper = False
            
            if self.use_transformer_emotion:
                try:
                    from transformers import pipeline
                    import torch
                    print("  Loading Transformers Emotion...")
                    self.emotion_model = pipeline(
                        "audio-classification",
                        model="superb/hubert-large-superb-er",
                        device=-1  # CPU mode
                    )
                    print("  ✅ Transformers Emotion loaded successfully")
                except Exception as e:
                    print(f"  ⚠️ Transformers Emotion not available: {e}")
                    self.use_transformer_emotion = False
            
            self.models_loaded = True
            print("✅ All models loaded")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def analyze_with_whisper(self, audio_file: str, language: str = 'en') -> Dict:
        """
        Analyze audio using Whisper transcription
        
        Args:
            audio_file: Path to audio file
            language: Language code ('en' for English, 'he' for Hebrew)
        """
        if not self.use_whisper or not self.whisper_model:
            return {}
        
        try:
            import os
            # Ensure absolute path and file exists
            audio_path = os.path.abspath(audio_file)
            if not os.path.exists(audio_path):
                print(f"⚠️ Audio file not found: {audio_path}")
                return {'error': f'File not found: {audio_path}'}
            
            # Normalize path for Windows
            audio_path = os.path.normpath(audio_path)
            
            # Map language code to Whisper language code
            whisper_lang = 'he' if language == 'he' else 'en'
            
            # Try to transcribe with better error handling
            try:
                # Use the same approach as inappropriate_language_detector
                result = self.whisper_model.transcribe(audio_path, language=whisper_lang, fp16=False)
            except FileNotFoundError as e:
                # This might be ffmpeg not found
                print(f"⚠️ Whisper file error (might be missing ffmpeg): {e}")
                print(f"   Audio path: {audio_path}")
                print(f"   File exists: {os.path.exists(audio_path)}")
                print(f"   Note: Whisper requires ffmpeg. Install it or use librosa preprocessing.")
                # Try alternative: load with librosa first
                try:
                    import librosa
                    print("   Trying alternative: loading audio with librosa first...")
                    audio_data, sr = librosa.load(audio_path, sr=16000)
                    # Save to temporary wav file
                    import tempfile
                    import soundfile as sf
                    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    tmp_path = tmp_file.name
                    tmp_file.close()  # Close file handle before writing
                    sf.write(tmp_path, audio_data, sr)
                    result = self.whisper_model.transcribe(tmp_path, language=whisper_lang, fp16=False)
                    os.unlink(tmp_path)  # Clean up
                    print("   ✅ Alternative method worked!")
                except Exception as e2:
                    print(f"   ⚠️ Alternative method also failed: {e2}")
                    return {'error': f'Whisper file error: {str(e)}. Note: ffmpeg might be required.'}
            except Exception as e:
                print(f"⚠️ Whisper transcription error: {e}")
                print(f"   Error type: {type(e).__name__}")
                return {'error': f'Whisper error: {str(e)}'}
            
            # Extract concerning patterns
            transcript = result["text"]
            segments = result["segments"]
            
            # Concerning keywords (language-specific)
            if language == 'he':
                concerning_keywords = {
                    'negative': ['לא', 'די', 'מספיק', 'עצור', 'אל', 'תפסיק', 'רגע'],
                    'emotional': ['רע', 'נורא', 'גרוע', 'מציק', 'מעצבן', 'כועס'],
                    'urgent': ['מהר', 'עכשיו', 'מיידי', 'דחוף']
                }
            else:  # English
                concerning_keywords = {
                    'negative': ['no', 'stop', 'enough', 'don\'t', 'quit', 'wait'],
                    'emotional': ['bad', 'terrible', 'awful', 'annoying', 'angry', 'mad'],
                    'urgent': ['hurry', 'now', 'immediate', 'urgent']
                }
            
            concerning_segments = []
            for seg in segments:
                text_lower = seg['text'].lower()
                categories_found = []
                
                for category, keywords in concerning_keywords.items():
                    if any(kw in text_lower for kw in keywords):
                        categories_found.append(category)
                
                if categories_found:
                    concerning_segments.append({
                        'start_time': seg['start'],
                        'end_time': seg['end'],
                        'text': seg['text'],
                        'categories': categories_found,
                        'no_speech_prob': seg.get('no_speech_prob', 0)
                    })
            
            return {
                'transcript': transcript,
                'num_segments': len(segments),
                'concerning_segments': concerning_segments,
                'transcript_length': len(transcript),
                'has_concerning_content': len(concerning_segments) > 0
            }
            
        except FileNotFoundError as e:
            import os
            print(f"⚠️ Whisper file not found error: {e}")
            print(f"   Audio file path: {audio_file}")
            print(f"   Absolute path: {os.path.abspath(audio_file) if audio_file else 'N/A'}")
            print(f"   File exists: {os.path.exists(audio_file) if audio_file else False}")
            print(f"   This might be a missing dependency (e.g., ffmpeg) required by Whisper")
            return {'error': f'File not found: {str(e)}', 'details': 'This might be a missing dependency (e.g., ffmpeg)'}
        except Exception as e:
            print(f"⚠️ Error in Whisper analysis: {e}")
            print(f"   Error type: {type(e).__name__}")
            return {'error': str(e), 'error_type': type(e).__name__}
    
    def analyze_emotions_advanced(self, audio_file: str, segments: List[tuple] = None) -> Dict:
        """
        Analyze emotions using advanced transformer models
        """
        if not self.use_transformer_emotion or not self.emotion_model:
            return {}
        
        try:
            import os
            # Ensure absolute path and file exists
            audio_path = os.path.abspath(audio_file)
            if not os.path.exists(audio_path):
                print(f"⚠️ Audio file not found: {audio_path}")
                return {'error': f'File not found: {audio_path}'}
            
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process in chunks
            chunk_duration = 30  # seconds
            chunk_samples = chunk_duration * sr
            chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)][:5]
            
            emotion_analysis = []
            for i, chunk in enumerate(chunks):
                try:
                    result = self.emotion_model(chunk)
                    emotion_analysis.append({
                        'chunk_index': i,
                        'start_time': i * chunk_duration,
                        'end_time': (i + 1) * chunk_duration,
                        'emotions': result[:3] if isinstance(result, list) else []
                    })
                except:
                    continue
            
            # Aggregate emotions
            emotion_scores = {}
            for analysis in emotion_analysis:
                for emo in analysis.get('emotions', []):
                    label = emo.get('label', 'unknown')
                    score = emo.get('score', 0)
                    if label not in emotion_scores:
                        emotion_scores[label] = []
                    emotion_scores[label].append(score)
            
            avg_scores = {k: np.mean(v) for k, v in emotion_scores.items()}
            top_emotions = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Detect concerning emotions
            concerning_emotions = ['anger', 'fear', 'sadness', 'disgust']
            concerning_detected = [
                (emo, score) for emo, score in top_emotions 
                if any(conc in emo.lower() for conc in concerning_emotions)
            ]
            
            return {
                'top_emotions': dict(top_emotions),
                'concerning_emotions': dict(concerning_detected),
                'chunks_analyzed': len(emotion_analysis),
                'has_concerning_emotions': len(concerning_detected) > 0,
                'details': emotion_analysis
            }
            
        except Exception as e:
            print(f"Error in advanced emotion analysis: {e}")
            return {'error': str(e)}
    
    def comprehensive_analysis(self, audio_file: str, language: str = 'en') -> Dict:
        """
        Perform comprehensive analysis using all advanced models
        
        Args:
            audio_file: Path to audio file
            language: Language code ('en' for English, 'he' for Hebrew)
        """
        results = {
            'whisper_analysis': {},
            'emotion_analysis': {},
            'combined_insights': {}
        }
        
        # Whisper analysis
        if self.use_whisper:
            print("\n📝 Running Whisper transcription...")
            results['whisper_analysis'] = self.analyze_with_whisper(audio_file, language=language)
        
        # Advanced emotion analysis
        if self.use_transformer_emotion:
            print("\n😊 Running advanced emotion analysis...")
            results['emotion_analysis'] = self.analyze_emotions_advanced(audio_file)
        
        # Combine insights
        results['combined_insights'] = self._combine_insights(results)
        
        return results
    
    def _combine_insights(self, results: Dict) -> Dict:
        """
        Combine insights from different models
        """
        insights = {
            'risk_score': 0.0,
            'concerns': [],
            'confidence': 0.0
        }
        
        # Analyze Whisper results
        whisper = results.get('whisper_analysis', {})
        if whisper.get('has_concerning_content'):
            insights['risk_score'] += 0.4
            insights['concerns'].append({
                'source': 'whisper_transcription',
                'type': 'concerning_language',
                'count': len(whisper.get('concerning_segments', []))
            })
        
        # Analyze emotion results
        emotions = results.get('emotion_analysis', {})
        if emotions.get('has_concerning_emotions'):
            insights['risk_score'] += 0.5
            insights['concerns'].append({
                'source': 'advanced_emotion_detection',
                'type': 'negative_emotions',
                'emotions': emotions.get('concerning_emotions', {})
            })
        
        # Calculate confidence
        num_sources = sum([
            1 if whisper else 0,
            1 if emotions else 0
        ])
        if num_sources > 0:
            insights['confidence'] = insights['risk_score'] / num_sources
        
        # Cap risk score
        insights['risk_score'] = min(insights['risk_score'], 1.0)
        
        return insights

if __name__ == "__main__":
    print("מנתח מתקדם עם מודלים מתקדמים")
    print("Advanced Analyzer with State-of-the-Art Models")
