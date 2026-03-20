"""
Violence Detection Module for Kindergarten Recording Analyzer
מודול זיהוי אלימות למערכת ניתוח הקלטות גן ילדים
"""

import logging
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)

class ViolenceDetector:
    def __init__(self):
        """
        Initialize Violence Detector
        אתחול מזהה אלימות
        """
        cfg = config.violence
        # Violence detection parameters
        # פרמטרים לזיהוי אלימות
        self.violence_thresholds = {
            'shouting': {
                'energy_threshold': cfg.shouting_energy,
                'frequency_variance_threshold': cfg.shouting_freq_variance,
                'spectral_rolloff_threshold': cfg.shouting_spectral_rolloff,
                'duration_threshold': cfg.shouting_duration
            },
            'aggressive_tone': {
                'energy_threshold': cfg.aggressive_energy,
                'pitch_variance_threshold': cfg.aggressive_pitch_variance,
                'spectral_bandwidth_threshold': cfg.aggressive_spectral_bandwidth,
                'zero_crossing_rate_threshold': cfg.aggressive_zcr
            },
            'threatening': {
                'energy_threshold': cfg.threatening_energy,
                'frequency_low_threshold': cfg.threatening_freq_low,
                'spectral_contrast_threshold': cfg.threatening_spectral_contrast,
                'duration_threshold': cfg.threatening_duration
            },
            'physical_violence_indicators': {
                'sudden_energy_spike_threshold': cfg.physical_energy_spike,
                'high_frequency_content_threshold': cfg.physical_hf_content,
                'rapid_changes_threshold': cfg.physical_rapid_changes
            }
        }

        # Context analysis parameters
        # פרמטרים לניתוח הקשר
        self.context_analysis = {
            'before_violence_window': cfg.before_violence_window,
            'after_violence_window': cfg.after_violence_window,
            'silence_after_violence_threshold': cfg.silence_after_violence,
            'continued_distress_threshold': cfg.continued_distress
        }
    
    def detect_violence_segments(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect potential violence in audio segments
        זיהוי אלימות פוטנציאלית בקטעי אודיו
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            List of detected violence segments with analysis
        """
        violence_segments = []
        
        # Analyze in 1-second segments with 0.5-second overlap
        segment_length = 1.0
        hop_length = 0.5
        segment_samples = int(segment_length * sr)
        hop_samples = int(hop_length * sr)
        
        for start_sample in range(0, len(audio) - segment_samples, hop_samples):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            
            # Calculate violence features
            features = self._calculate_violence_features(segment, sr)
            
            # Detect different types of violence
            violence_types = self._detect_violence_types(features)
            
            if violence_types:
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                violence_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'features': features,
                    'violence_types': violence_types,
                    'severity': self._calculate_severity(features, violence_types),
                    'confidence': self._calculate_confidence(features, violence_types)
                })
        
        # Merge nearby violence segments and analyze context
        merged_segments = self._merge_violence_segments(violence_segments)
        context_analyzed = self._analyze_violence_context(audio, sr, merged_segments)
        
        return context_analyzed
    
    def _calculate_violence_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Calculate features specific to violence detection
        חישוב תכונות ספציפיות לזיהוי אלימות
        
        Args:
            audio: Audio segment
            sr: Sample rate
            
        Returns:
            Dictionary of violence-related features
        """
        features = {}
        
        # Energy features (violence often involves high energy)
        rms_energy = librosa.feature.rms(y=audio)[0]
        features['mean_energy'] = np.mean(rms_energy)
        features['max_energy'] = np.max(rms_energy)
        features['energy_variance'] = np.var(rms_energy)
        features['energy_skewness'] = self._calculate_skewness(rms_energy)
        
        # Frequency features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['mean_frequency'] = np.mean(spectral_centroid)
        features['frequency_variance'] = np.var(spectral_centroid)
        features['frequency_range'] = np.max(spectral_centroid) - np.min(spectral_centroid)
        
        # Spectral features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['mean_spectral_rolloff'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_variance'] = np.var(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['mean_spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['mean_spectral_contrast'] = np.mean(spectral_contrast, axis=1)
        
        # Pitch and voice characteristics
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        pitch_variance = np.var(chroma, axis=1)
        features['mean_pitch_variance'] = np.mean(pitch_variance)
        
        # Zero crossing rate (speech vs noise)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['mean_zero_crossing_rate'] = np.mean(zcr)
        features['zcr_variance'] = np.var(zcr)
        
        # MFCC features for voice characteristics
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_variance'] = np.var(mfccs, axis=1)
        
        # Temporal dynamics (rapid changes indicate violence)
        features['temporal_instability'] = self._calculate_temporal_instability(audio, sr)
        
        # High frequency content (screams, sharp sounds)
        features['high_frequency_ratio'] = self._calculate_high_frequency_ratio(audio, sr)
        
        return features
    
    # Minimum absolute energy to consider any violence detection.
    # Below this level the audio is too quiet for any violence type.
    MIN_ENERGY_GATE = 0.05

    def _detect_violence_types(self, features: Dict) -> List[str]:
        """
        Detect specific types of violence based on features
        זיהוי סוגים ספציפיים של אלימות בהתבסס על תכונות
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            List of detected violence types
        """
        # B1 FIX: energy gate -- reject all violence for quiet audio
        if features['mean_energy'] < self.MIN_ENERGY_GATE:
            return []

        detected_types = []
        
        # Check for shouting
        shouting_thresholds = self.violence_thresholds['shouting']
        if (features['mean_energy'] > shouting_thresholds['energy_threshold'] and
            features['frequency_variance'] > shouting_thresholds['frequency_variance_threshold'] and
            features['mean_spectral_rolloff'] > shouting_thresholds['spectral_rolloff_threshold']):
            detected_types.append('shouting')
        
        # Check for aggressive tone
        aggressive_thresholds = self.violence_thresholds['aggressive_tone']
        if (features['mean_energy'] > aggressive_thresholds['energy_threshold'] and
            features['mean_pitch_variance'] > aggressive_thresholds['pitch_variance_threshold'] and
            features['mean_spectral_bandwidth'] > aggressive_thresholds['spectral_bandwidth_threshold']):
            detected_types.append('aggressive_tone')
        
        # Check for threatening behavior (low pitch, sustained)
        threatening_thresholds = self.violence_thresholds['threatening']
        if (features['mean_energy'] > threatening_thresholds['energy_threshold'] and
            features['mean_frequency'] < threatening_thresholds['frequency_low_threshold'] and
            features['mean_spectral_contrast'][0] > threatening_thresholds['spectral_contrast_threshold']):
            detected_types.append('threatening')
        
        # Check for physical violence indicators
        # B1 FIX: require energy gate AND at least 2 of 3 conditions
        physical_thresholds = self.violence_thresholds['physical_violence_indicators']
        physical_hits = sum([
            features['max_energy'] > physical_thresholds['sudden_energy_spike_threshold'],
            features['high_frequency_ratio'] > physical_thresholds['high_frequency_content_threshold'],
            features['temporal_instability'] > physical_thresholds['rapid_changes_threshold'],
        ])
        if physical_hits >= 2:
            detected_types.append('potential_physical_violence')
        
        return detected_types
    
    def _calculate_severity(self, features: Dict, violence_types: List[str]) -> str:
        """
        Calculate severity level of detected violence
        חישוב רמת חומרה של אלימות שזוהתה
        
        Args:
            features: Dictionary of audio features
            violence_types: List of detected violence types
            
        Returns:
            Severity level (low, medium, high, critical)
        """
        severity_scores = []
        
        # Base severity by type
        type_severity = {
            'shouting': 2,
            'aggressive_tone': 3,
            'threatening': 4,
            'potential_physical_violence': 5
        }
        
        for violence_type in violence_types:
            base_score = type_severity.get(violence_type, 1)
            
            # Adjust based on intensity
            if features['mean_energy'] > 0.3:
                base_score += 1
            if features['max_energy'] > 0.4:
                base_score += 1
            if features['temporal_instability'] > 0.7:
                base_score += 1
            
            severity_scores.append(min(base_score, 5))  # Cap at 5
        
        max_severity = max(severity_scores) if severity_scores else 0
        
        severity_map = {1: 'low', 2: 'low', 3: 'medium', 4: 'high', 5: 'critical'}
        return severity_map.get(max_severity, 'low')
    
    def _calculate_confidence(self, features: Dict, violence_types: List[str]) -> float:
        """
        Calculate confidence score for violence detection
        חישוב ציון ביטחון לזיהוי אלימות
        
        Args:
            features: Dictionary of audio features
            violence_types: List of detected violence types
            
        Returns:
            Confidence score between 0 and 1
        """
        if not violence_types:
            return 0.0
        
        confidence_factors = []
        
        # Energy confidence
        energy_confidence = min(features['mean_energy'] / 0.3, 1.0)
        confidence_factors.append(energy_confidence * 0.3)
        
        # Frequency variance confidence
        freq_confidence = min(features['frequency_variance'] / 0.5, 1.0)
        confidence_factors.append(freq_confidence * 0.2)
        
        # Temporal instability confidence
        temporal_confidence = min(features['temporal_instability'] / 0.8, 1.0)
        confidence_factors.append(temporal_confidence * 0.2)
        
        # Multiple violence types increase confidence
        type_confidence = min(len(violence_types) / 3.0, 1.0)
        confidence_factors.append(type_confidence * 0.3)
        
        return sum(confidence_factors)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calculate skewness of data
        חישוב חוסר סימטריה של נתונים
        """
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_temporal_instability(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate temporal instability (rapid changes)
        חישוב חוסר יציבות זמנית (שינויים מהירים)
        """
        # Calculate short-time energy
        frame_length = 1024
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate rate of change
        energy_diff = np.abs(np.diff(energy))
        return np.mean(energy_diff) if len(energy_diff) > 0 else 0
    
    def _calculate_high_frequency_ratio(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate ratio of high frequency content
        חישוב יחס תוכן תדרים גבוהים
        """
        # Calculate spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Define high frequency threshold (above 2kHz)
        high_freq_threshold = 2000
        high_freq_ratio = np.sum(spectral_centroid > high_freq_threshold) / len(spectral_centroid)
        
        return high_freq_ratio
    
    MERGE_GAP_SECONDS = 1.5
    MAX_MERGED_DURATION_SECONDS = 20.0

    def _merge_violence_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge nearby violence segments
        מיזוג קטעי אלימות קרובים
        
        Args:
            segments: List of violence segments
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            would_merge_duration = max(last['end_time'], current['end_time']) - last['start_time']
            gap_ok = current['start_time'] <= last['end_time'] + self.MERGE_GAP_SECONDS
            duration_ok = would_merge_duration <= self.MAX_MERGED_DURATION_SECONDS

            if gap_ok and duration_ok:
                merged[-1] = {
                    'start_time': last['start_time'],
                    'end_time': max(last['end_time'], current['end_time']),
                    'duration': would_merge_duration,
                    'features': current['features'],
                    'violence_types': list(set(last['violence_types'] + current['violence_types'])),
                    'severity': max(last['severity'], current['severity'], key=lambda x: ['low', 'medium', 'high', 'critical'].index(x)),
                    'confidence': max(last['confidence'], current['confidence'])
                }
            else:
                merged.append(current)
        
        return merged
    
    def _analyze_violence_context(self, audio: np.ndarray, sr: int, violence_segments: List[Dict]) -> List[Dict]:
        """
        Analyze context around violence incidents
        ניתוח הקשר סביב אירועי אלימות
        
        Args:
            audio: Full audio data
            sr: Sample rate
            violence_segments: List of violence segments
            
        Returns:
            List of violence segments with context analysis
        """
        enhanced_segments = []
        
        for segment in violence_segments:
            context_analysis = self._analyze_single_violence_context(audio, sr, segment)
            
            enhanced_segment = segment.copy()
            enhanced_segment['context'] = context_analysis
            
            # Adjust severity based on context
            enhanced_segment['adjusted_severity'] = self._adjust_severity_by_context(
                segment['severity'], context_analysis
            )
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _analyze_single_violence_context(self, audio: np.ndarray, sr: int, segment: Dict) -> Dict:
        """
        Analyze context for a single violence segment
        ניתוח הקשר עבור קטע אלימות יחיד
        
        Args:
            audio: Full audio data
            sr: Sample rate
            segment: Violence segment
            
        Returns:
            Context analysis results
        """
        context = {
            'before_violence': {},
            'after_violence': {},
            'overall_assessment': 'unknown'
        }
        
        # Analyze period before violence
        before_start = max(0, segment['start_time'] - self.context_analysis['before_violence_window'])
        before_end = segment['start_time']
        before_audio = audio[int(before_start * sr):int(before_end * sr)]
        
        if len(before_audio) > 0:
            context['before_violence'] = self._analyze_context_period(before_audio, sr)
        
        # Analyze period after violence
        after_start = segment['end_time']
        after_end = min(len(audio) / sr, segment['end_time'] + self.context_analysis['after_violence_window'])
        after_audio = audio[int(after_start * sr):int(after_end * sr)]
        
        if len(after_audio) > 0:
            context['after_violence'] = self._analyze_context_period(after_audio, sr)
        
        # Overall assessment
        context['overall_assessment'] = self._assess_overall_context(context)
        
        return context
    
    def _analyze_context_period(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Analyze a specific time period for context
        ניתוח תקופה ספציפית בזמן להקשר
        
        Args:
            audio: Audio data for the period
            sr: Sample rate
            
        Returns:
            Context analysis for the period
        """
        if len(audio) < sr * 0.1:  # Too short for analysis
            return {'has_activity': False, 'activity_type': 'unknown', 'intensity': 'low'}
        
        # Calculate basic features
        rms_energy = librosa.feature.rms(y=audio)[0]
        mean_energy = np.mean(rms_energy)
        
        # Determine activity level
        has_activity = mean_energy > 0.05
        
        if not has_activity:
            return {'has_activity': False, 'activity_type': 'silence', 'intensity': 'low'}
        
        # Analyze activity type
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_frequency = np.mean(spectral_centroid)
        
        if mean_frequency > 400:  # High frequency (likely crying)
            activity_type = 'distress'
        elif mean_frequency < 200:  # Low frequency (likely adult voice)
            activity_type = 'adult_speech'
        else:
            activity_type = 'mixed'
        
        # Determine intensity
        if mean_energy < 0.1:
            intensity = 'low'
        elif mean_energy < 0.2:
            intensity = 'medium'
        else:
            intensity = 'high'
        
        return {
            'has_activity': True,
            'activity_type': activity_type,
            'intensity': intensity,
            'mean_energy': mean_energy,
            'mean_frequency': mean_frequency
        }
    
    def _assess_overall_context(self, context: Dict) -> str:
        """
        Assess overall context of violence incident
        הערכת הקשר כללי של אירוע אלימות
        
        Args:
            context: Context analysis results
            
        Returns:
            Overall context assessment
        """
        before = context.get('before_violence', {})
        after = context.get('after_violence', {})
        
        # Check for escalation pattern
        if (before.get('activity_type') == 'distress' and 
            before.get('intensity') == 'high' and
            after.get('activity_type') == 'silence'):
            return 'escalation_then_silence'
        
        # Check for continued distress
        if (after.get('activity_type') == 'distress' and 
            after.get('intensity') in ['medium', 'high']):
            return 'continued_distress'
        
        # Check for immediate response
        if after.get('activity_type') == 'adult_speech':
            return 'immediate_adult_response'
        
        # Check for sudden outburst
        if before.get('activity_type') == 'silence':
            return 'sudden_outburst'
        
        return 'unclear_context'
    
    def _adjust_severity_by_context(self, base_severity: str, context: Dict) -> str:
        """
        Adjust severity based on context analysis
        התאמת חומרה בהתבסס על ניתוח הקשר
        
        Args:
            base_severity: Base severity level
            context: Context analysis results
            
        Returns:
            Adjusted severity level
        """
        severity_levels = ['low', 'medium', 'high', 'critical']
        base_index = severity_levels.index(base_severity)
        
        overall_assessment = context.get('overall_assessment', 'unknown')
        
        # Adjust based on context
        if overall_assessment == 'escalation_then_silence':
            base_index = min(base_index + 1, len(severity_levels) - 1)
        elif overall_assessment == 'continued_distress':
            base_index = min(base_index + 1, len(severity_levels) - 1)
        elif overall_assessment == 'immediate_adult_response':
            base_index = max(base_index - 1, 0)
        
        return severity_levels[base_index]

if __name__ == "__main__":
    # Test the violence detector
    detector = ViolenceDetector()
    print("Violence Detector initialized successfully")
    print("Ready to detect violence in kindergarten recordings...")
