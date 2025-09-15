"""
Baby Cry Detection Module for Kindergarten Recording Analyzer
מודול זיהוי בכי תינוקות למערכת ניתוח הקלטות גן ילדים
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CryDetector:
    def __init__(self):
        """
        Initialize Baby Cry Detector
        אתחול מזהה בכי תינוקות
        """
        # Baby cry characteristics
        # מאפיינים של בכי תינוקות
        self.cry_features = {
            'frequency_range': (200, 800),  # Hz - typical baby cry frequency range
            'duration_range': (0.5, 10.0),  # seconds - typical cry duration
            'energy_threshold': 0.08,  # minimum energy level for cry detection
            'pitch_variance_threshold': 0.15,  # pitch variation in cries
            'spectral_rolloff_threshold': 0.4,  # high frequency content
            'zero_crossing_rate_threshold': 0.02  # speech-like characteristics
        }
        
        # Response detection parameters
        # פרמטרים לזיהוי תגובה
        self.response_features = {
            'response_window': 10.0,  # seconds - window to look for responses after cry
            'min_response_duration': 1.0,  # minimum response duration
            'response_energy_threshold': 0.05,  # minimum energy for response
            'response_pitch_threshold': 100  # Hz - typical adult voice pitch
        }
    
    def detect_cry_segments(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect baby cry segments in audio
        זיהוי קטעי בכי תינוקות באודיו
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            List of detected cry segments with metadata
        """
        cry_segments = []
        
        # Segment audio for analysis
        segment_length = 2.0  # 2-second segments
        hop_length = 0.5  # 0.5-second overlap
        segment_samples = int(segment_length * sr)
        hop_samples = int(hop_length * sr)
        
        for start_sample in range(0, len(audio) - segment_samples, hop_samples):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            
            # Calculate features for this segment
            features = self._calculate_cry_features(segment, sr)
            
            # Check if this segment contains a cry
            is_cry = self._is_cry_segment(features)
            
            if is_cry:
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                cry_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'features': features,
                    'confidence': self._calculate_cry_confidence(features),
                    'intensity': self._calculate_cry_intensity(features)
                })
        
        # Merge overlapping cry segments
        merged_segments = self._merge_overlapping_segments(cry_segments)
        
        return merged_segments
    
    def _calculate_cry_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Calculate features specific to baby cry detection
        חישוב תכונות ספציפיות לזיהוי בכי תינוקות
        
        Args:
            audio: Audio segment
            sr: Sample rate
            
        Returns:
            Dictionary of cry-specific features
        """
        features = {}
        
        # Energy features
        rms_energy = librosa.feature.rms(y=audio)[0]
        features['mean_energy'] = np.mean(rms_energy)
        features['max_energy'] = np.max(rms_energy)
        features['energy_variance'] = np.var(rms_energy)
        
        # Frequency features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['mean_frequency'] = np.mean(spectral_centroid)
        features['frequency_variance'] = np.var(spectral_centroid)
        
        # Check if frequency is in typical cry range
        cry_freq_min, cry_freq_max = self.cry_features['frequency_range']
        features['in_cry_frequency_range'] = cry_freq_min <= features['mean_frequency'] <= cry_freq_max
        
        # Spectral rolloff (high frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['mean_spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['mean_zero_crossing_rate'] = np.mean(zcr)
        
        # Pitch features (using chroma as proxy)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        pitch_variance = np.var(chroma, axis=1)
        features['mean_pitch_variance'] = np.mean(pitch_variance)
        
        # MFCC features for vocal characteristics
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        
        # Spectral contrast for timbre
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        
        return features
    
    def _is_cry_segment(self, features: Dict) -> bool:
        """
        Determine if a segment contains a baby cry
        קביעה אם קטע מכיל בכי תינוק
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            True if segment likely contains a cry
        """
        score = 0
        max_score = 0
        
        # Energy check
        if features['mean_energy'] > self.cry_features['energy_threshold']:
            score += 2
        max_score += 2
        
        # Frequency range check
        if features['in_cry_frequency_range']:
            score += 3
        max_score += 3
        
        # Pitch variance (cries have variable pitch)
        if features['mean_pitch_variance'] > self.cry_features['pitch_variance_threshold']:
            score += 2
        max_score += 2
        
        # Spectral rolloff (cries have high frequency content)
        if features['mean_spectral_rolloff'] > self.cry_features['spectral_rolloff_threshold']:
            score += 1
        max_score += 1
        
        # Zero crossing rate (not too high for cries)
        if features['mean_zero_crossing_rate'] < self.cry_features['zero_crossing_rate_threshold']:
            score += 1
        max_score += 1
        
        # Duration check (basic - will be refined in merging)
        if 0.5 <= 2.0 <= 10.0:  # segment length is 2 seconds
            score += 1
        max_score += 1
        
        # Threshold for cry detection
        return (score / max_score) >= 0.6
    
    def _calculate_cry_confidence(self, features: Dict) -> float:
        """
        Calculate confidence score for cry detection
        חישוב ציון ביטחון לזיהוי בכי
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0
        max_score = 0
        
        # Energy confidence
        energy_ratio = min(features['mean_energy'] / (self.cry_features['energy_threshold'] * 2), 1.0)
        score += energy_ratio * 2
        max_score += 2
        
        # Frequency confidence
        if features['in_cry_frequency_range']:
            freq_min, freq_max = self.cry_features['frequency_range']
            freq_center = (freq_min + freq_max) / 2
            freq_distance = abs(features['mean_frequency'] - freq_center) / freq_center
            freq_confidence = max(0, 1 - freq_distance)
            score += freq_confidence * 3
        max_score += 3
        
        # Pitch variance confidence
        pitch_confidence = min(features['mean_pitch_variance'] / (self.cry_features['pitch_variance_threshold'] * 2), 1.0)
        score += pitch_confidence * 2
        max_score += 2
        
        return score / max_score if max_score > 0 else 0
    
    def _calculate_cry_intensity(self, features: Dict) -> str:
        """
        Calculate cry intensity level
        חישוב רמת עוצמת בכי
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Intensity level (low, medium, high)
        """
        energy = features['mean_energy']
        
        if energy < 0.12:
            return 'low'
        elif energy < 0.2:
            return 'medium'
        else:
            return 'high'
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge overlapping cry segments
        מיזוג קטעי בכי חופפים
        
        Args:
            segments: List of cry segments
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            # Check if segments overlap or are close (within 1 second)
            if current['start_time'] <= last['end_time'] + 1.0:
                # Merge segments
                merged[-1] = {
                    'start_time': last['start_time'],
                    'end_time': max(last['end_time'], current['end_time']),
                    'duration': max(last['end_time'], current['end_time']) - last['start_time'],
                    'features': current['features'],  # Use more recent features
                    'confidence': max(last['confidence'], current['confidence']),
                    'intensity': current['intensity'] if current['intensity'] != 'low' else last['intensity']
                }
            else:
                merged.append(current)
        
        return merged
    
    def detect_response_to_cry(self, audio: np.ndarray, sr: int, cry_segments: List[Dict]) -> List[Dict]:
        """
        Detect staff responses to baby cries
        זיהוי תגובות הצוות לבכי תינוקות
        
        Args:
            audio: Full audio data
            sr: Sample rate
            cry_segments: List of detected cry segments
            
        Returns:
            List of cry segments with response analysis
        """
        cry_with_responses = []
        
        for cry in cry_segments:
            response_window = self.response_features['response_window']
            response_start = cry['end_time']
            response_end = min(response_start + response_window, len(audio) / sr)
            
            # Extract response window
            start_sample = int(response_start * sr)
            end_sample = int(response_end * sr)
            response_audio = audio[start_sample:end_sample]
            
            # Analyze response
            response_analysis = self._analyze_response_segment(response_audio, sr)
            
            cry_with_response = cry.copy()
            cry_with_response['response_analysis'] = response_analysis
            cry_with_response['response_detected'] = response_analysis['has_response']
            
            if response_analysis['has_response']:
                cry_with_response['response_start'] = response_analysis['response_start'] + response_start
                cry_with_response['response_end'] = response_analysis['response_end'] + response_start
                cry_with_response['response_quality'] = response_analysis['response_quality']
            
            cry_with_responses.append(cry_with_response)
        
        return cry_with_responses
    
    def _analyze_response_segment(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Analyze a segment for staff response to cry
        ניתוח קטע לתגובת צוות לבכי
        
        Args:
            audio: Audio segment to analyze
            sr: Sample rate
            
        Returns:
            Response analysis results
        """
        if len(audio) < sr * 0.5:  # Too short for meaningful analysis
            return {
                'has_response': False,
                'response_start': 0,
                'response_end': 0,
                'response_quality': 'none',
                'confidence': 0.0
            }
        
        # Look for speech-like activity
        segment_length = 1.0  # 1-second analysis windows
        segment_samples = int(segment_length * sr)
        response_segments = []
        
        for start_sample in range(0, len(audio) - segment_samples, segment_samples // 2):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            
            # Calculate response features
            features = self._calculate_response_features(segment, sr)
            
            if self._is_response_segment(features):
                response_segments.append({
                    'start_time': start_sample / sr,
                    'end_time': end_sample / sr,
                    'features': features,
                    'confidence': self._calculate_response_confidence(features)
                })
        
        # Merge response segments
        merged_responses = self._merge_overlapping_segments(response_segments) if response_segments else []
        
        if merged_responses:
            best_response = max(merged_responses, key=lambda x: x['confidence'])
            return {
                'has_response': True,
                'response_start': best_response['start_time'],
                'response_end': best_response['end_time'],
                'response_quality': self._assess_response_quality(best_response['features']),
                'confidence': best_response['confidence']
            }
        else:
            return {
                'has_response': False,
                'response_start': 0,
                'response_end': 0,
                'response_quality': 'none',
                'confidence': 0.0
            }
    
    def _calculate_response_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Calculate features for response detection
        חישוב תכונות לזיהוי תגובה
        
        Args:
            audio: Audio segment
            sr: Sample rate
            
        Returns:
            Dictionary of response features
        """
        features = {}
        
        # Energy features
        rms_energy = librosa.feature.rms(y=audio)[0]
        features['mean_energy'] = np.mean(rms_energy)
        
        # Frequency features (adult voice typically lower than baby cry)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['mean_frequency'] = np.mean(spectral_centroid)
        
        # Zero crossing rate (speech characteristics)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['mean_zero_crossing_rate'] = np.mean(zcr)
        
        # MFCC features for speech recognition
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_variance'] = np.var(mfccs, axis=1)
        
        return features
    
    def _is_response_segment(self, features: Dict) -> bool:
        """
        Determine if segment contains a response
        קביעה אם קטע מכיל תגובה
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            True if segment likely contains a response
        """
        # Check for speech-like characteristics
        has_energy = features['mean_energy'] > self.response_features['response_energy_threshold']
        has_speech_frequency = features['mean_frequency'] < self.response_features['response_pitch_threshold']
        has_speech_zcr = 0.01 < features['mean_zero_crossing_rate'] < 0.1
        
        return has_energy and has_speech_frequency and has_speech_zcr
    
    def _calculate_response_confidence(self, features: Dict) -> float:
        """
        Calculate confidence for response detection
        חישוב ביטחון לזיהוי תגובה
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0
        max_score = 0
        
        # Energy confidence
        energy_ratio = min(features['mean_energy'] / (self.response_features['response_energy_threshold'] * 3), 1.0)
        score += energy_ratio * 2
        max_score += 2
        
        # Frequency confidence (adult voice)
        if features['mean_frequency'] < self.response_features['response_pitch_threshold']:
            freq_confidence = 1 - (features['mean_frequency'] / self.response_features['response_pitch_threshold'])
            score += freq_confidence * 2
        max_score += 2
        
        # ZCR confidence (speech-like)
        zcr = features['mean_zero_crossing_rate']
        if 0.01 < zcr < 0.1:
            zcr_confidence = 1 - abs(zcr - 0.05) / 0.05
            score += max(0, zcr_confidence)
        max_score += 1
        
        return score / max_score if max_score > 0 else 0
    
    def _assess_response_quality(self, features: Dict) -> str:
        """
        Assess quality of response
        הערכת איכות התגובה
        
        Args:
            features: Dictionary of response features
            
        Returns:
            Response quality (poor, adequate, good)
        """
        energy = features['mean_energy']
        
        if energy < 0.08:
            return 'poor'
        elif energy < 0.15:
            return 'adequate'
        else:
            return 'good'

if __name__ == "__main__":
    # Test the cry detector
    detector = CryDetector()
    print("Baby Cry Detector initialized successfully")
    print("Ready to detect baby cries and staff responses...")
