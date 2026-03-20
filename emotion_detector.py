"""
Emotion Detection Module for SoundShield-AI

This module provides emotion detection capabilities for analyzing kindergarten
audio recordings to identify concerning emotions such as anger, stress, and
aggression in staff voices.

מודול זיהוי רגשות למערכת ניתוח הקלטות גן ילדים
"""

import logging
from typing import Dict, List, Optional
import warnings

import numpy as np
import librosa

warnings.filterwarnings('ignore')

from config import config

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_EMOTIONS = ['anger', 'stress', 'calm', 'aggression']
DEFAULT_CONFIDENCE_THRESHOLD = config.emotion.confidence_threshold
MIN_SEGMENT_LENGTH_SECONDS = config.emotion.min_segment_length


class EmotionDetectionError(Exception):
    """Base exception for emotion detection errors."""
    pass

class EmotionDetector:
    """Detects emotions in audio using acoustic feature analysis.
    
    This class analyzes audio segments to detect emotional states including
    anger, stress, aggression, and calm. It uses acoustic features such as
    energy, pitch variance, spectral characteristics, and MFCCs.
    
    Attributes:
        emotion_thresholds: Dictionary of threshold values for each emotion
        
    Example:
        >>> detector = EmotionDetector()
        >>> audio = np.random.randn(16000)  # 1 second at 16kHz
        >>> features = detector.calculate_emotion_features(audio, 16000)
        >>> emotion = detector.detect_emotion(features)
        >>> print(emotion['primary_emotion'])
    """
    
    def __init__(self):
        """Initialize Emotion Detector with threshold configurations."""
        logger.info("Initializing Emotion Detector")
        
        cfg = config.emotion
        # Emotion thresholds based on audio features
        # ספי רגשות בהתבסס על תכונות אודיו
        self.emotion_thresholds = {
            'anger': {
                'energy_threshold': cfg.anger_energy,
                'pitch_variance_threshold': cfg.anger_pitch_variance,
                'spectral_rolloff_threshold': cfg.anger_spectral_rolloff
            },
            'stress': {
                'energy_variance_threshold': cfg.stress_energy_variance,
                'zero_crossing_rate_threshold': cfg.stress_zcr,
                'spectral_centroid_variance_threshold': cfg.stress_spectral_centroid_variance
            },
            'calm': {
                'energy_threshold': cfg.calm_energy,
                'pitch_stability_threshold': cfg.calm_pitch_stability,
                'spectral_rolloff_threshold': cfg.calm_spectral_rolloff
            },
            'aggression': {
                'energy_threshold': cfg.aggression_energy,
                'pitch_variance_threshold': cfg.aggression_pitch_variance,
                'spectral_bandwidth_threshold': cfg.aggression_spectral_bandwidth
            }
        }
        
        logger.info("Emotion Detector initialized successfully")
    
    def calculate_emotion_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Calculate emotion-related audio features.
        
        Extracts acoustic features that correlate with emotional states,
        including energy, pitch, spectral characteristics, and MFCCs.
        
        חישוב תכונות אודיו הקשורות לרגשות
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate in Hz
            
        Returns:
            Dictionary containing extracted emotion features
            
        Raises:
            EmotionDetectionError: If feature extraction fails
            ValueError: If audio is too short or invalid
            
        Example:
            >>> detector = EmotionDetector()
            >>> audio = np.random.randn(16000)
            >>> features = detector.calculate_emotion_features(audio, 16000)
            >>> print(features.keys())
        """
        if len(audio) == 0:
            raise ValueError("Audio array is empty")
        
        if sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")
        
        if len(audio) / sr < MIN_SEGMENT_LENGTH_SECONDS:
            raise ValueError(
                f"Audio too short: {len(audio)/sr:.2f}s "
                f"(minimum: {MIN_SEGMENT_LENGTH_SECONDS}s)"
            )
        
        try:
            features = {}
            
            # Energy features
            rms_energy = librosa.feature.rms(y=audio)[0]
            features['mean_energy'] = float(np.mean(rms_energy))
            features['energy_variance'] = float(np.var(rms_energy))
            features['energy_max'] = float(np.max(rms_energy))
            
            # Pitch features (using chroma as proxy for pitch)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            pitch_variance = np.var(chroma, axis=1)
            features['mean_pitch_variance'] = float(np.mean(pitch_variance))
            features['pitch_stability'] = float(1.0 / (1.0 + np.mean(pitch_variance)))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['mean_spectral_centroid'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_variance'] = float(np.var(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['mean_spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['mean_spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
            # Zero crossing rate (indicates speech vs noise)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['mean_zero_crossing_rate'] = float(np.mean(zcr))
            features['zero_crossing_rate_variance'] = float(np.var(zcr))
            
            # MFCC features for speech characteristics
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_variance'] = np.var(mfccs, axis=1).tolist()
            
            logger.debug(f"Extracted {len(features)} emotion features")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise EmotionDetectionError(f"Failed to extract emotion features: {e}")
    
    def detect_emotion(self, features: Dict) -> Dict:
        """Detect emotion based on audio features.
        
        Analyzes extracted features to classify emotional state using
        threshold-based scoring system.
        
        זיהוי רגש בהתבסס על תכונות אודיו
        
        Args:
            features: Dictionary of audio features from calculate_emotion_features
            
        Returns:
            Dictionary with emotion scores, primary emotion, and confidence:
            {
                'emotion_scores': Dict[str, float],
                'primary_emotion': str,
                'confidence': float
            }
            
        Example:
            >>> detector = EmotionDetector()
            >>> features = {'mean_energy': 0.2, 'mean_pitch_variance': 0.4, ...}
            >>> result = detector.detect_emotion(features)
            >>> print(f"{result['primary_emotion']}: {result['confidence']:.2f}")
        """
        if not features:
            raise ValueError("Features dictionary is empty")
        
        emotion_scores = {}
        
        # Calculate scores for each emotion
        for emotion, thresholds in self.emotion_thresholds.items():
            score = 0.0
            weight_sum = 0.0
            
            try:
                if emotion == 'anger':
                    # High energy + high pitch variance + high spectral rolloff = anger
                    if features['mean_energy'] > thresholds['energy_threshold']:
                        score += 0.4
                    weight_sum += 0.4
                    
                    if features['mean_pitch_variance'] > thresholds['pitch_variance_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                    
                    if features['mean_spectral_rolloff'] > thresholds['spectral_rolloff_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                
                elif emotion == 'stress':
                    # High energy variance + high ZCR + high spectral centroid variance = stress
                    if features['energy_variance'] > thresholds['energy_variance_threshold']:
                        score += 0.4
                    weight_sum += 0.4
                    
                    if features['mean_zero_crossing_rate'] > thresholds['zero_crossing_rate_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                    
                    if features['spectral_centroid_variance'] > thresholds['spectral_centroid_variance_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                
                elif emotion == 'calm':
                    # Low energy + high pitch stability + low spectral rolloff = calm
                    if features['mean_energy'] < thresholds['energy_threshold']:
                        score += 0.4
                    weight_sum += 0.4
                    
                    if features['pitch_stability'] > thresholds['pitch_stability_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                    
                    if features['mean_spectral_rolloff'] < thresholds['spectral_rolloff_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                
                elif emotion == 'aggression':
                    # Very high energy + very high pitch variance + high spectral bandwidth = aggression
                    if features['mean_energy'] > thresholds['energy_threshold']:
                        score += 0.4
                    weight_sum += 0.4
                    
                    if features['mean_pitch_variance'] > thresholds['pitch_variance_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                    
                    if features['mean_spectral_bandwidth'] > thresholds['spectral_bandwidth_threshold']:
                        score += 0.3
                    weight_sum += 0.3
                
                # Normalize score
                emotion_scores[emotion] = score / weight_sum if weight_sum > 0 else 0.0
                
            except KeyError as e:
                logger.warning(f"Missing feature for {emotion} detection: {e}")
                emotion_scores[emotion] = 0.0
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
        else:
            primary_emotion = 'calm'
            confidence = 0.0
        
        logger.debug(f"Detected emotion: {primary_emotion} (confidence: {confidence:.2f})")
        
        return {
            'emotion_scores': emotion_scores,
            'primary_emotion': primary_emotion,
            'confidence': confidence
        }
    
    def analyze_segment_emotions(self, audio_segments: List[np.ndarray], sr: int) -> List[Dict]:
        """
        Analyze emotions for multiple audio segments
        ניתוח רגשות עבור מספר קטעי אודיו
        
        Args:
            audio_segments: List of audio segments
            sr: Sample rate
            
        Returns:
            List of emotion analysis results for each segment
        """
        segment_emotions = []
        
        for i, segment in enumerate(audio_segments):
            if len(segment) < sr * 0.5:  # Skip segments shorter than 0.5 seconds
                continue
                
            features = self.calculate_emotion_features(segment, sr)
            emotion_result = self.detect_emotion(features)
            
            segment_emotions.append({
                'segment_index': i,
                'start_time': i * 5.0,  # Assuming 5-second segments
                'end_time': (i + 1) * 5.0,
                'features': features,
                'emotion_analysis': emotion_result
            })
        
        return segment_emotions
    
    def detect_concerning_emotions(self, segment_emotions: List[Dict], threshold: float = 0.6) -> List[Dict]:
        """
        Detect segments with concerning emotions (anger, stress, aggression)
        זיהוי קטעים עם רגשות מדאיגים (כעס, לחץ, אגרסיה)
        
        Args:
            segment_emotions: List of emotion analysis results
            threshold: Confidence threshold for concerning emotions
            
        Returns:
            List of concerning segments with details
        """
        concerning_segments = []
        concerning_emotions = ['anger', 'stress', 'aggression']
        
        for segment in segment_emotions:
            emotion_analysis = segment['emotion_analysis']
            primary_emotion = emotion_analysis['primary_emotion']
            confidence = emotion_analysis['confidence']
            
            if primary_emotion in concerning_emotions and confidence > threshold:
                concerning_segments.append({
                    'segment_index': segment['segment_index'],
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'detected_emotion': primary_emotion,
                    'confidence': confidence,
                    'all_scores': emotion_analysis['emotion_scores'],
                    'severity': self._calculate_severity(primary_emotion, confidence)
                })
        
        return concerning_segments
    
    def merge_with_advanced_results(self, heuristic_results: List[Dict],
                                     advanced_results: List[Dict]) -> List[Dict]:
        """Merge heuristic emotion results with HuBERT advanced results.

        HuBERT confidence > 0.7 -> trust HuBERT
        HuBERT confidence < 0.5 -> weighted average of both
        Between 0.5-0.7 -> prefer HuBERT but blend confidence

        Returns unified format expected by report_generator.py.
        """
        if not advanced_results:
            for r in heuristic_results:
                r['ml_backed'] = False
            return heuristic_results

        if not heuristic_results:
            return advanced_results

        # Index advanced results by time range for matching
        merged = []
        used_advanced = set()

        for h in heuristic_results:
            h_start = h.get('start_time', 0)
            h_end = h.get('end_time', 0)

            # Find overlapping advanced result
            best_match = None
            best_overlap = 0
            for j, a in enumerate(advanced_results):
                a_start = a.get('start_time', 0)
                a_end = a.get('end_time', 0)
                overlap = max(0, min(h_end, a_end) - max(h_start, a_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (j, a)

            if best_match:
                j, a = best_match
                used_advanced.add(j)
                adv_conf = a.get('confidence', 0)

                if adv_conf > 0.7:
                    # Trust HuBERT completely
                    merged.append(a)
                elif adv_conf < 0.5:
                    # Weighted average - prefer heuristic slightly
                    blended_conf = 0.4 * adv_conf + 0.6 * h.get('confidence', 0)
                    result = dict(h)
                    result['confidence'] = blended_conf
                    result['ml_backed'] = False
                    merged.append(result)
                else:
                    # Blend: prefer HuBERT emotion but adjust confidence
                    blended_conf = 0.6 * adv_conf + 0.4 * h.get('confidence', 0)
                    result = dict(a)
                    result['confidence'] = blended_conf
                    merged.append(result)
            else:
                h_copy = dict(h)
                h_copy['ml_backed'] = False
                merged.append(h_copy)

        # Add any advanced results that didn't match heuristic ones
        for j, a in enumerate(advanced_results):
            if j not in used_advanced:
                merged.append(a)

        return merged

    def _calculate_severity(self, emotion: str, confidence: float) -> str:
        """
        Calculate severity level for detected emotion
        חישוב רמת חומרה עבור רגש שזוהה
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            
        Returns:
            Severity level (low, medium, high, critical)
        """
        base_severity = {
            'anger': 'medium',
            'stress': 'low',
            'aggression': 'high'
        }
        
        severity_map = {
            'low': ['low', 'medium'],
            'medium': ['medium', 'high'],
            'high': ['high', 'critical']
        }
        
        base = base_severity.get(emotion, 'low')
        
        if confidence > 0.8:
            return severity_map[base][1]  # Higher severity
        else:
            return severity_map[base][0]  # Lower severity

if __name__ == "__main__":
    # Test the emotion detector
    detector = EmotionDetector()
    print("Emotion Detector initialized successfully")
    print("Ready to detect emotions in kindergarten recordings...")
