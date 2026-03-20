"""
Advanced Analyzer with State-of-the-Art Models

Supports Faster-Whisper (CTranslate2) for 4x speedup,
falling back to standard OpenAI Whisper if unavailable.
"""

import logging
import numpy as np
import librosa
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)


class AdvancedAnalyzer:
    """
    Advanced analyzer using best-in-class models.

    Whisper priority: faster-whisper > openai-whisper > disabled.
    """

    # HuBERT emotion label mapping to severity schema
    HUBERT_LABEL_MAP = {
        'ang': {'emotion': 'anger', 'base_severity': 'high'},
        'hap': {'emotion': 'calm', 'base_severity': 'low'},
        'sad': {'emotion': 'stress', 'base_severity': 'medium'},
        'neu': {'emotion': 'calm', 'base_severity': 'low'},
    }

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
        self.whisper_loaded = False
        self.hubert_loaded = False
        self._whisper_backend = None  # 'faster-whisper' or 'openai-whisper'

        logger.info("Initializing Advanced Analyzer...")
        
    def load_models(self):
        """Load advanced models if available.

        Whisper loading priority:
        1. faster-whisper (CTranslate2) — 4x faster, 75% less memory
        2. openai-whisper — standard fallback
        3. disabled
        """
        model_name = config.advanced.whisper_model
        hubert_model_name = config.advanced.hubert_model

        try:
            if self.use_whisper:
                # Try faster-whisper first
                try:
                    from faster_whisper import WhisperModel
                    logger.info(f"  Loading Faster-Whisper ({model_name})...")
                    self.whisper_model = WhisperModel(
                        model_name, device="cpu", compute_type="int8"
                    )
                    self._whisper_backend = 'faster-whisper'
                    self.whisper_loaded = True
                    logger.info("  Faster-Whisper loaded successfully")
                except ImportError:
                    # Fallback to standard whisper
                    try:
                        import whisper
                        logger.info(f"  Loading OpenAI Whisper ({model_name})...")
                        self.whisper_model = whisper.load_model(model_name)
                        self._whisper_backend = 'openai-whisper'
                        self.whisper_loaded = True
                        logger.info("  OpenAI Whisper loaded successfully")
                    except ImportError:
                        logger.warning("  Whisper not installed - skipping")
                        self.use_whisper = False
                    except Exception as e:
                        logger.warning(f"  Error loading Whisper: {e}")
                        self.use_whisper = False
                except Exception as e:
                    logger.warning(f"  Error loading Faster-Whisper: {e}")
                    # Fallback to standard whisper
                    try:
                        import whisper
                        self.whisper_model = whisper.load_model(model_name)
                        self._whisper_backend = 'openai-whisper'
                        self.whisper_loaded = True
                        logger.info("  Fell back to OpenAI Whisper")
                    except Exception:
                        self.use_whisper = False

            if self.use_transformer_emotion:
                # Priority: ONNX Runtime > PyTorch HuBERT > disabled
                onnx_loaded = False
                try:
                    from model_optimizer import optimizer
                    if optimizer.load_onnx_model():
                        self.emotion_model = optimizer
                        self.hubert_loaded = True
                        self._emotion_backend = 'onnx'
                        onnx_loaded = True
                        logger.info("  HuBERT loaded via ONNX Runtime")
                except Exception:
                    pass

                if not onnx_loaded:
                    try:
                        from transformers import pipeline
                        import torch
                        logger.info(f"  Loading HuBERT Emotion Model ({hubert_model_name})...")
                        self.emotion_model = pipeline(
                            "audio-classification",
                            model=hubert_model_name,
                            device=-1  # CPU mode
                        )
                        self.hubert_loaded = True
                        self._emotion_backend = 'pytorch'
                        logger.info("  HuBERT Emotion loaded successfully (PyTorch)")
                    except Exception as e:
                        logger.warning(f"  HuBERT Emotion not available: {e}")
                        self.use_transformer_emotion = False

            self.models_loaded = self.whisper_loaded or self.hubert_loaded
            logger.info(f"Models loaded (whisper={self.whisper_loaded} "
                        f"[{self._whisper_backend}], hubert={self.hubert_loaded} "
                        f"[{getattr(self, '_emotion_backend', 'none')}])")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
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
            audio_path = os.path.abspath(audio_file)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return {'error': f'File not found: {audio_path}'}

            audio_path = os.path.normpath(audio_path)
            whisper_lang = 'he' if language == 'he' else 'en'

            # --- Transcribe using the loaded backend ---
            try:
                if self._whisper_backend == 'faster-whisper':
                    result = self._transcribe_faster_whisper(audio_path, whisper_lang)
                else:
                    result = self._transcribe_openai_whisper(audio_path, whisper_lang)
            except FileNotFoundError as e:
                logger.warning(f"Whisper file error (might be missing ffmpeg): {e}")
                # Fallback: pre-process with librosa
                try:
                    import tempfile
                    import soundfile as sf
                    logger.info("Trying librosa pre-processing fallback...")
                    audio_data, sr = librosa.load(audio_path, sr=16000)
                    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    tmp_path = tmp_file.name
                    tmp_file.close()
                    sf.write(tmp_path, audio_data, sr)
                    if self._whisper_backend == 'faster-whisper':
                        result = self._transcribe_faster_whisper(tmp_path, whisper_lang)
                    else:
                        result = self._transcribe_openai_whisper(tmp_path, whisper_lang)
                    os.unlink(tmp_path)
                except Exception as e2:
                    logger.error(f"Librosa fallback also failed: {e2}")
                    return {'error': f'Whisper file error: {str(e)}'}
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
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
            
        except Exception as e:
            logger.error(f"Error in Whisper analysis: {e}")
            return {'error': str(e), 'error_type': type(e).__name__}
    
    def _transcribe_faster_whisper(self, audio_path: str, language: str) -> Dict:
        """Transcribe using faster-whisper (CTranslate2 backend)."""
        segments_iter, info = self.whisper_model.transcribe(
            audio_path, language=language, beam_size=5
        )
        segments_list = []
        full_text_parts = []
        for seg in segments_iter:
            segments_list.append({
                'start': seg.start,
                'end': seg.end,
                'text': seg.text,
                'no_speech_prob': seg.no_speech_prob,
            })
            full_text_parts.append(seg.text)
        return {
            'text': ' '.join(full_text_parts),
            'segments': segments_list,
        }

    def _transcribe_openai_whisper(self, audio_path: str, language: str) -> Dict:
        """Transcribe using standard OpenAI Whisper."""
        result = self.whisper_model.transcribe(audio_path, language=language, fp16=False)
        return {
            'text': result['text'],
            'segments': [
                {
                    'start': s['start'],
                    'end': s['end'],
                    'text': s['text'],
                    'no_speech_prob': s.get('no_speech_prob', 0),
                }
                for s in result['segments']
            ],
        }

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
    
    def detect_concerning_emotions_advanced(self, audio_file: str) -> List[Dict]:
        """
        Detect concerning emotions using HuBERT in 5-10s chunks.
        Returns results in the same schema as EmotionDetector.detect_concerning_emotions().

        Each result dict has:
            segment_index, start_time, end_time, detected_emotion, confidence,
            severity, all_scores, ml_backed
        """
        if not self.hubert_loaded or not self.emotion_model:
            return []

        try:
            import os
            audio_path = os.path.abspath(audio_file)
            if not os.path.exists(audio_path):
                return []

            audio, sr = librosa.load(audio_path, sr=16000)

            chunk_duration = 7  # seconds (between 5-10s)
            chunk_samples = chunk_duration * sr
            chunks = [audio[i:i + chunk_samples]
                      for i in range(0, len(audio), chunk_samples)]

            concerning = ['anger', 'stress', 'aggression']
            results = []

            for i, chunk in enumerate(chunks):
                if len(chunk) < sr * 0.5:  # skip very short chunks
                    continue
                try:
                    predictions = self.emotion_model(chunk)
                    if not isinstance(predictions, list):
                        continue

                    # Map HuBERT labels to our schema
                    all_scores = {}
                    primary_emotion = 'calm'
                    primary_score = 0.0

                    for pred in predictions:
                        label = pred.get('label', '').lower()
                        score = pred.get('score', 0.0)
                        mapped = self.HUBERT_LABEL_MAP.get(label)
                        if mapped:
                            emotion_name = mapped['emotion']
                            all_scores[emotion_name] = max(
                                all_scores.get(emotion_name, 0.0), score
                            )
                            if score > primary_score:
                                primary_score = score
                                primary_emotion = emotion_name

                    start_time = i * chunk_duration
                    end_time = min((i + 1) * chunk_duration, len(audio) / sr)

                    if primary_emotion in concerning and primary_score > 0.5:
                        severity = self._map_severity(primary_emotion, primary_score)
                        results.append({
                            'segment_index': i,
                            'start_time': float(start_time),
                            'end_time': float(end_time),
                            'detected_emotion': primary_emotion,
                            'confidence': float(primary_score),
                            'severity': severity,
                            'all_scores': all_scores,
                            'ml_backed': True
                        })
                except Exception:
                    continue

            return results

        except Exception as e:
            print(f"Error in detect_concerning_emotions_advanced: {e}")
            return []

    def _map_severity(self, emotion: str, confidence: float) -> str:
        """Map emotion + confidence to severity level."""
        base_map = {
            'anger': 'medium',
            'stress': 'low',
            'aggression': 'high',
        }
        severity_upgrade = {
            'low': ['low', 'medium'],
            'medium': ['medium', 'high'],
            'high': ['high', 'critical'],
        }
        base = base_map.get(emotion, 'low')
        levels = severity_upgrade.get(base, ['low', 'medium'])
        return levels[1] if confidence > 0.8 else levels[0]

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
