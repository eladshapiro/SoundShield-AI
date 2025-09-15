"""
Audio Analysis Module for Kindergarten Recording Analyzer
מודול ניתוח אודיו למערכת ניתוח הקלטות גן ילדים
"""

import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AudioAnalyzer:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Audio Analyzer
        אתחול מנתח האודיו
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate
        טעינת קובץ אודיו והחזרת נתוני אודיו וקצב דגימה
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract comprehensive audio features
        חילוץ תכונות מקיפות של האודיו
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic audio features
        features['duration'] = len(audio) / sr
        features['sample_rate'] = sr
        features['rms_energy'] = librosa.feature.rms(y=audio)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfccs'] = mfccs
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Tonnetz features
        features['tonnetz'] = librosa.feature.tonnetz(y=audio, sr=sr)
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        return features
    
    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> List[Tuple[float, float]]:
        """
        Detect silent segments in audio
        זיהוי קטעים שקטים באודיו
        
        Args:
            audio: Audio data
            threshold: Silence threshold
            
        Returns:
            List of (start_time, end_time) tuples for silent segments
        """
        # Calculate RMS energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find silent frames
        silent_frames = rms < threshold
        
        # Convert frames to time
        times = librosa.frames_to_time(np.arange(len(silent_frames)), sr=self.sample_rate, hop_length=hop_length)
        
        # Find continuous silent segments
        silent_segments = []
        start_time = None
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and start_time is None:
                start_time = times[i]
            elif not is_silent and start_time is not None:
                silent_segments.append((start_time, times[i]))
                start_time = None
        
        # Handle case where audio ends during silence
        if start_time is not None:
            silent_segments.append((start_time, times[-1]))
        
        return silent_segments
    
    def detect_loud_segments(self, audio: np.ndarray, threshold: float = 0.3) -> List[Tuple[float, float]]:
        """
        Detect loud segments (potential crying, shouting)
        זיהוי קטעים רועשים (בכי פוטנציאלי, צעקות)
        
        Args:
            audio: Audio data
            threshold: Loudness threshold
            
        Returns:
            List of (start_time, end_time) tuples for loud segments
        """
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find loud frames
        loud_frames = rms > threshold
        
        # Convert frames to time
        times = librosa.frames_to_time(np.arange(len(loud_frames)), sr=self.sample_rate, hop_length=hop_length)
        
        # Find continuous loud segments
        loud_segments = []
        start_time = None
        
        for i, is_loud in enumerate(loud_frames):
            if is_loud and start_time is None:
                start_time = times[i]
            elif not is_loud and start_time is not None:
                loud_segments.append((start_time, times[i]))
                start_time = None
        
        if start_time is not None:
            loud_segments.append((start_time, times[-1]))
        
        return loud_segments
    
    def segment_audio(self, audio: np.ndarray, segment_length: float = 5.0) -> List[np.ndarray]:
        """
        Segment audio into smaller chunks for analysis
        חלוקת האודיו לחתיכות קטנות יותר לניתוח
        
        Args:
            audio: Audio data
            segment_length: Length of each segment in seconds
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * self.sample_rate)
        segments = []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) > segment_samples // 2:  # Only include segments with at least half the desired length
                segments.append(segment)
        
        return segments
    
    def analyze_audio_file(self, file_path: str) -> Dict:
        """
        Complete audio analysis of a file
        ניתוח אודיו מלא של קובץ
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"Loading audio file: {file_path}")
        audio, sr = self.load_audio(file_path)
        
        print("Extracting features...")
        features = self.extract_features(audio, sr)
        
        print("Detecting silence...")
        silent_segments = self.detect_silence(audio)
        
        print("Detecting loud segments...")
        loud_segments = self.detect_loud_segments(audio)
        
        print("Segmenting audio...")
        segments = self.segment_audio(audio)
        
        analysis_result = {
            'file_path': file_path,
            'duration': features['duration'],
            'features': features,
            'silent_segments': silent_segments,
            'loud_segments': loud_segments,
            'num_segments': len(segments),
            'segments': segments,
            'sample_rate': sr
        }
        
        print(f"Analysis complete. Duration: {features['duration']:.2f} seconds")
        print(f"Found {len(silent_segments)} silent segments and {len(loud_segments)} loud segments")
        
        return analysis_result

if __name__ == "__main__":
    # Test the audio analyzer
    analyzer = AudioAnalyzer()
    print("Audio Analyzer initialized successfully")
    print("Ready to analyze kindergarten recordings...")
