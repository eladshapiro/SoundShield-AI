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

TARGET_SR = 16000


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
        self._whisper_models: Dict[str, object] = {}   # language -> faster-whisper model
        self._whisper_device = 'cpu'
        self.transcriptions: Dict[tuple, Dict] = {}     # (abs path, language) -> {'text', 'segments'}
        self.emotion_analyzer = None                    # emotion_models.EmotionAnalyzer
        self.dim_emotion_loaded = False

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
                # faster-whisper models are loaded lazily per language (see _get_whisper);
                # here we only verify the backend and warm the English model.
                try:
                    import faster_whisper  # noqa: F401
                    self._whisper_backend = 'faster-whisper'
                    self.whisper_loaded = True
                    logger.info("  Faster-Whisper backend available "
                                f"(en={config.advanced.whisper_model_for('en')}, "
                                f"he={config.advanced.whisper_model_for('he')})")
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
                        device = 0 if torch.cuda.is_available() else -1
                        logger.info(f"  Loading HuBERT Emotion Model ({hubert_model_name}) on "
                                    f"{'cuda:0' if device == 0 else 'cpu'}...")
                        self.emotion_model = pipeline(
                            "audio-classification",
                            model=hubert_model_name,
                            device=device
                        )
                        self.hubert_loaded = True
                        self._emotion_backend = 'pytorch'
                        logger.info("  HuBERT Emotion loaded successfully (PyTorch)")
                    except Exception as e:
                        logger.warning(f"  HuBERT Emotion not available: {e}")
                        self.use_transformer_emotion = False

            if self.use_transformer_emotion:
                try:
                    from emotion_models import EmotionAnalyzer, CategoricalEmotionModel
                    categorical = None
                    if self.hubert_loaded and getattr(self, '_emotion_backend', '') == 'pytorch':
                        categorical = CategoricalEmotionModel()
                        categorical.pipe = self.emotion_model
                        categorical.loaded = True
                    self.emotion_analyzer = EmotionAnalyzer(
                        use_dimensional=config.advanced.use_dimensional_emotion,
                        use_categorical=self.hubert_loaded,
                        categorical=categorical,
                    )
                    if self.emotion_analyzer.dim is not None:
                        self.dim_emotion_loaded = self.emotion_analyzer.dim.load()
                except Exception as e:
                    logger.warning(f"  Emotion analyzer unavailable: {e}")

            self.models_loaded = self.whisper_loaded or self.hubert_loaded or self.dim_emotion_loaded
            logger.info(f"Models loaded (whisper={self.whisper_loaded} "
                        f"[{self._whisper_backend}], hubert={self.hubert_loaded} "
                        f"[{getattr(self, '_emotion_backend', 'none')}], "
                        f"dimensional_emotion={self.dim_emotion_loaded})")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _get_whisper(self, language: str):
        """Load (once) and return the faster-whisper model for ``language``."""
        if self._whisper_backend != 'faster-whisper':
            return self.whisper_model
        if language in self._whisper_models:
            return self._whisper_models[language]
        from faster_whisper import WhisperModel
        from cuda_compat import preload_cuda12_libs, cuda_available
        cfg = config.advanced
        model_name = cfg.whisper_model_for(language)
        device = cfg.whisper_device
        if device == 'auto':
            device = 'cuda' if cuda_available() else 'cpu'
        compute_type = cfg.whisper_compute_type
        if compute_type == 'auto':
            compute_type = 'float16' if device == 'cuda' else 'int8'
        if device == 'cuda':
            preload_cuda12_libs()
        logger.info(f"  Loading Faster-Whisper {model_name} for '{language}' on {device} ({compute_type})")
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            if device == 'cuda':
                logger.warning(f"  CUDA load failed ({e}); falling back to CPU int8")
                device, compute_type = 'cpu', 'int8'
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            else:
                raise
        self._whisper_device = device
        self._whisper_models[language] = model
        self.whisper_model = model
        return model

    def transcribe(self, audio_file: str, language: str = 'en') -> Dict:
        """Transcribe once per (file, language); result is cached and shared with
        the inappropriate-language detector. Returns {'text', 'segments'}."""
        import os
        audio_path = os.path.normpath(os.path.abspath(audio_file))
        key = (audio_path, language)
        if key in self.transcriptions:
            return self.transcriptions[key]
        whisper_lang = 'he' if language == 'he' else 'en'
        if self._whisper_backend == 'faster-whisper':
            self._get_whisper(whisper_lang)
            result = self._transcribe_faster_whisper(audio_path, whisper_lang)
        else:
            result = self._transcribe_openai_whisper(audio_path, whisper_lang)
        self.transcriptions[key] = result
        return result

    def analyze_with_whisper(self, audio_file: str, language: str = 'en') -> Dict:
        """
        Analyze audio using Whisper transcription
        
        Args:
            audio_file: Path to audio file
            language: Language code ('en' for English, 'he' for Hebrew)
        """
        # ``whisper_model`` is only populated lazily by ``_get_whisper`` on the
        # faster-whisper path, so gate on ``whisper_loaded`` (set by load_models)
        # or the report's transcript section stays empty on the first analysis.
        if not self.use_whisper or not self.whisper_loaded:
            return {}

        try:
            import os
            audio_path = os.path.abspath(audio_file)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return {'error': f'File not found: {audio_path}'}

            audio_path = os.path.normpath(audio_path)
            whisper_lang = 'he' if language == 'he' else 'en'

            # --- Transcribe using the loaded backend (cached per file+language) ---
            try:
                result = self.transcribe(audio_path, language)
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
                    result = self.transcribe(tmp_path, language)
                    self.transcriptions[(audio_path, language)] = result
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
                'segments': segments,
                'num_segments': len(segments),
                'concerning_segments': concerning_segments,
                'transcript_length': len(transcript),
                'has_concerning_content': len(concerning_segments) > 0,
                'model': (config.advanced.whisper_model_for(whisper_lang)
                          if self._whisper_backend == 'faster-whisper' else config.advanced.whisper_model),
                'device': self._whisper_device,
            }
            
        except Exception as e:
            logger.error(f"Error in Whisper analysis: {e}")
            return {'error': str(e), 'error_type': type(e).__name__}
    
    def _transcribe_faster_whisper(self, audio_path: str, language: str) -> Dict:
        """Transcribe using faster-whisper (CTranslate2 backend)."""
        model = self._get_whisper(language)
        segments_iter, info = model.transcribe(
            audio_path, language=language, beam_size=config.advanced.whisper_beam_size,
            vad_filter=True,
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

    # ------------------------------------------------------------------
    # Emotion (neural): windowed HuBERT + arousal/dominance timelines
    # ------------------------------------------------------------------
    def emotion_timeline(self, audio_file: str = None, audio: np.ndarray = None, sr: int = None):
        """Compute (or return None) the windowed emotion timeline for a file or array."""
        if self.emotion_analyzer is None:
            return None
        try:
            if audio is None:
                import os
                audio_path = os.path.abspath(audio_file)
                if not os.path.exists(audio_path):
                    return None
                audio, sr = librosa.load(audio_path, sr=TARGET_SR)
            return self.emotion_analyzer.timeline(audio, sr)
        except Exception as e:
            logger.warning(f"Emotion timeline failed: {e}")
            return None

    def analyze_emotions_advanced(self, audio_file: str, segments: List[tuple] = None,
                                  emotion_timeline=None) -> Dict:
        """Recording-level emotion summary from the windowed timeline."""
        if not self.use_transformer_emotion:
            return {}
        try:
            tl = emotion_timeline if emotion_timeline is not None else self.emotion_timeline(audio_file)
            if tl is None:
                return {}
            top: Dict[str, float] = {}
            for label, arr in tl.categorical.items():
                top[self.HUBERT_LABEL_MAP.get(label, {}).get('emotion', label)] = float(np.mean(arr))
            if tl.has_dimensional:
                top['arousal'] = float(np.mean(tl.arousal))
                top['dominance'] = float(np.mean(tl.dominance))
                top['valence'] = float(np.mean(tl.valence))
            concerning = self.concerning_emotions_from_timeline(tl)
            concerning_summary: Dict[str, float] = {}
            for c in concerning:
                concerning_summary[c['detected_emotion']] = max(
                    concerning_summary.get(c['detected_emotion'], 0.0), c['confidence'])
            return {
                'top_emotions': dict(sorted(top.items(), key=lambda x: x[1], reverse=True)),
                'concerning_emotions': concerning_summary,
                'chunks_analyzed': int(len(tl.starts)),
                'has_concerning_emotions': bool(concerning),
                'details': tl.to_dict(),
            }
        except Exception as e:
            logger.warning(f"Error in advanced emotion analysis: {e}")
            return {'error': str(e)}

    def concerning_emotions_from_timeline(self, emotion_timeline, tag_timeline=None) -> List[Dict]:
        """Concerning-emotion segments (schema of EmotionDetector.detect_concerning_emotions).

        * ``anger``      — HuBERT P(anger) >= anger_prob_threshold
        * ``aggression`` — arousal AND dominance above thresholds (dimensional model)
        Both are gated with the AudioSet tagger when available: the window must
        contain speech and must not be dominated by a crying child.
        """
        from emotion_models import anger_rule
        tcfg = config.tagger
        results: List[Dict] = []
        if emotion_timeline is None:
            return results
        p_ang = emotion_timeline.categorical.get('ang')
        for i, start in enumerate(emotion_timeline.starts):
            end = min(start + emotion_timeline.window, emotion_timeline.duration)
            if tag_timeline is not None:
                # speech must dominate any crying (a distressed child is not an angry adult)
                if tag_timeline.score('speech_dominant', start, end) < tcfg.speech_threshold:
                    continue
            anger = float(p_ang[i]) if p_ang is not None else 0.0
            if emotion_timeline.has_dimensional:
                flag, confidence, emotion = anger_rule(
                    anger, float(emotion_timeline.arousal[i]),
                    float(emotion_timeline.dominance[i]), float(emotion_timeline.valence[i]))
            else:
                flag, confidence, emotion = anger_rule(anger, None, None, None)
            if not flag:
                continue
            all_scores = {'anger': anger}
            if emotion_timeline.has_dimensional:
                all_scores.update({'arousal': float(emotion_timeline.arousal[i]),
                                   'dominance': float(emotion_timeline.dominance[i]),
                                   'valence': float(emotion_timeline.valence[i])})
            results.append({
                'segment_index': i,
                'start_time': float(start),
                'end_time': float(end),
                'detected_emotion': emotion,
                'confidence': float(confidence),
                'severity': self._map_severity(emotion, confidence),
                'all_scores': all_scores,
                'ml_backed': True,
            })
        merged = self._merge_adjacent_emotions(results)
        min_windows = config.advanced.emotion_min_windows
        if min_windows > 1:
            min_len = emotion_timeline.window + (min_windows - 1) * config.advanced.emotion_hop_seconds
            merged = [m for m in merged if m['end_time'] - m['start_time'] >= min_len - 1e-6
                      or m['end_time'] >= emotion_timeline.duration - 1e-6]
        return merged

    @staticmethod
    def _merge_adjacent_emotions(results: List[Dict]) -> List[Dict]:
        """Merge overlapping / touching windows with the same emotion into one segment."""
        merged: List[Dict] = []
        for r in sorted(results, key=lambda x: x['start_time']):
            if merged and merged[-1]['detected_emotion'] == r['detected_emotion'] \
                    and r['start_time'] <= merged[-1]['end_time']:
                last = merged[-1]
                last['end_time'] = max(last['end_time'], r['end_time'])
                if r['confidence'] > last['confidence']:
                    last['confidence'] = r['confidence']
                    last['severity'] = r['severity']
                    last['all_scores'] = r['all_scores']
            else:
                merged.append(dict(r))
        return merged

    def detect_concerning_emotions_advanced(self, audio_file: str, emotion_timeline=None,
                                            tag_timeline=None) -> List[Dict]:
        """
        Detect concerning emotions with the neural models.
        Returns results in the same schema as EmotionDetector.detect_concerning_emotions().
        """
        if self.emotion_analyzer is None:
            return []
        try:
            if emotion_timeline is None:
                emotion_timeline = self.emotion_timeline(audio_file)
            return self.concerning_emotions_from_timeline(emotion_timeline, tag_timeline)
        except Exception as e:
            logger.warning(f"Error in detect_concerning_emotions_advanced: {e}")
            return []

    def _map_severity(self, emotion: str, confidence: float) -> str:
        """Map emotion + confidence to severity level.

        ``anger`` means the categorical (HuBERT) and dimensional models agree;
        ``aggression`` is flagged by the dimensional model alone (high arousal
        and dominance, low valence) and is reported one level lower because
        excited speech can trigger it.
        """
        base_map = {
            'anger': 'medium',
            'stress': 'low',
            'aggression': 'low',
        }
        severity_upgrade = {
            'low': ['low', 'medium'],
            'medium': ['medium', 'high'],
            'high': ['high', 'critical'],
        }
        base = base_map.get(emotion, 'low')
        levels = severity_upgrade.get(base, ['low', 'medium'])
        return levels[1] if confidence > 0.8 else levels[0]

    def comprehensive_analysis(self, audio_file: str, language: str = 'en',
                               emotion_timeline=None) -> Dict:
        """
        Perform comprehensive analysis using all advanced models
        
        Args:
            audio_file: Path to audio file
            language: Language code ('en' for English, 'he' for Hebrew)
            emotion_timeline: optional precomputed EmotionTimeline (avoids re-running models)
        """
        results = {
            'whisper_analysis': {},
            'emotion_analysis': {},
            'combined_insights': {}
        }
        
        # Whisper analysis
        if self.use_whisper:
            logger.info("Running Whisper transcription...")
            results['whisper_analysis'] = self.analyze_with_whisper(audio_file, language=language)
        
        # Advanced emotion analysis
        if self.use_transformer_emotion:
            logger.info("Running advanced emotion analysis...")
            results['emotion_analysis'] = self.analyze_emotions_advanced(
                audio_file, emotion_timeline=emotion_timeline)
        
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
