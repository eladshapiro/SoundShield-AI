"""
Neglect Detection Module for Kindergarten Recording Analyzer
מודול זיהוי הזנחה למערכת ניתוח הקלטות גן ילדים
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NeglectDetector:
    def __init__(self):
        """
        Initialize Neglect Detector
        אתחול מזהה הזנחה
        """
        # Neglect detection parameters
        # פרמטרים לזיהוי הזנחה
        self.neglect_thresholds = {
            'unanswered_cry_duration': 30.0,  # seconds - cry without response
            'min_cry_duration': 5.0,  # minimum cry duration to be considered
            'response_window': 15.0,  # window to look for response after cry
            'silence_after_distress': 60.0,  # suspicious silence after distress
            'repeated_unanswered_cries': 3,  # number of unanswered cries to flag
            'long_term_silence': 300.0,  # 5 minutes of silence might indicate neglect
        }
        
        # Activity patterns that indicate neglect
        # דפוסי פעילות שמעידים על הזנחה
        self.neglect_patterns = {
            'prolonged_silence': {
                'duration_threshold': 120.0,  # 2 minutes
                'energy_threshold': 0.02
            },
            'lack_of_interaction': {
                'adult_speech_threshold': 0.1,  # minimum adult speech per hour
                'interaction_window': 3600.0  # 1 hour
            },
            'ignored_distress': {
                'distress_duration_threshold': 20.0,
                'response_energy_threshold': 0.08
            }
        }
    
    def detect_neglect_patterns(self, audio: np.ndarray, sr: int, 
                              cry_segments: List[Dict] = None, 
                              violence_segments: List[Dict] = None) -> Dict:
        """
        Detect neglect patterns in the audio
        זיהוי דפוסי הזנחה באודיו
        
        Args:
            audio: Audio data
            sr: Sample rate
            cry_segments: Previously detected cry segments
            violence_segments: Previously detected violence segments
            
        Returns:
            Dictionary containing neglect analysis results
        """
        neglect_analysis = {
            'unanswered_cries': [],
            'prolonged_silence_periods': [],
            'lack_of_interaction_periods': [],
            'ignored_distress_episodes': [],
            'overall_neglect_score': 0.0,
            'neglect_severity': 'none'
        }
        
        print("Analyzing for neglect patterns...")
        
        # Detect unanswered cries
        if cry_segments:
            print("Checking for unanswered cries...")
            neglect_analysis['unanswered_cries'] = self._detect_unanswered_cries(
                audio, sr, cry_segments
            )
        
        # Detect prolonged silence periods
        print("Checking for prolonged silence...")
        neglect_analysis['prolonged_silence_periods'] = self._detect_prolonged_silence(
            audio, sr
        )
        
        # Detect lack of interaction
        print("Checking for lack of interaction...")
        neglect_analysis['lack_of_interaction_periods'] = self._detect_lack_of_interaction(
            audio, sr
        )
        
        # Detect ignored distress
        print("Checking for ignored distress...")
        neglect_analysis['ignored_distress_episodes'] = self._detect_ignored_distress(
            audio, sr, violence_segments
        )
        
        # Calculate overall neglect score
        neglect_analysis['overall_neglect_score'] = self._calculate_neglect_score(neglect_analysis)
        neglect_analysis['neglect_severity'] = self._determine_neglect_severity(
            neglect_analysis['overall_neglect_score']
        )
        
        return neglect_analysis
    
    def _detect_unanswered_cries(self, audio: np.ndarray, sr: int, 
                                cry_segments: List[Dict]) -> List[Dict]:
        """
        Detect cries that went unanswered by staff
        זיהוי בכי שלא זכה לתגובת הצוות
        
        Args:
            audio: Audio data
            sr: Sample rate
            cry_segments: List of detected cry segments
            
        Returns:
            List of unanswered cry episodes
        """
        unanswered_cries = []
        
        for cry in cry_segments:
            # Check if cry was long enough to warrant a response
            if cry['duration'] < self.neglect_thresholds['min_cry_duration']:
                continue
            
            # Look for response in the response window
            response_window = self.neglect_thresholds['response_window']
            response_start = cry['end_time']
            response_end = min(response_start + response_window, len(audio) / sr)
            
            # Extract response period
            start_sample = int(response_start * sr)
            end_sample = int(response_end * sr)
            response_audio = audio[start_sample:end_sample]
            
            # Analyze response
            has_response = self._detect_staff_response(response_audio, sr)
            
            if not has_response:
                # Calculate neglect severity for this episode
                neglect_severity = self._calculate_cry_neglect_severity(cry)
                
                unanswered_cries.append({
                    'cry_start_time': cry['start_time'],
                    'cry_end_time': cry['end_time'],
                    'cry_duration': cry['duration'],
                    'cry_intensity': cry.get('intensity', 'medium'),
                    'response_window_start': response_start,
                    'response_window_end': response_end,
                    'neglect_severity': neglect_severity,
                    'time_without_response': cry['duration'] + response_window
                })
        
        return unanswered_cries
    
    def _detect_staff_response(self, response_audio: np.ndarray, sr: int) -> bool:
        """
        Detect if there was a staff response in the audio segment
        זיהוי אם הייתה תגובת צוות בקטע האודיו
        
        Args:
            response_audio: Audio segment to analyze
            sr: Sample rate
            
        Returns:
            True if staff response detected
        """
        if len(response_audio) < sr * 0.5:  # Too short for analysis
            return False
        
        # Calculate energy and frequency characteristics
        rms_energy = librosa.feature.rms(y=response_audio)[0]
        mean_energy = np.mean(rms_energy)
        
        if mean_energy < 0.05:  # Too quiet for speech
            return False
        
        # Analyze spectral characteristics for adult speech
        spectral_centroid = librosa.feature.spectral_centroid(y=response_audio, sr=sr)[0]
        mean_frequency = np.mean(spectral_centroid)
        
        # Adult speech typically has lower frequency than baby cries
        if mean_frequency < 400:  # Adult voice range
            # Additional check for speech-like characteristics
            zcr = librosa.feature.zero_crossing_rate(response_audio)[0]
            mean_zcr = np.mean(zcr)
            
            # Speech has specific zero-crossing rate characteristics
            if 0.01 < mean_zcr < 0.1:
                return True
        
        return False
    
    def _calculate_cry_neglect_severity(self, cry: Dict) -> str:
        """
        Calculate neglect severity for a specific cry episode
        חישוב חומרת הזנחה עבור אירוע בכי ספציפי
        
        Args:
            cry: Cry segment information
            
        Returns:
            Severity level (low, medium, high, critical)
        """
        duration = cry['duration']
        intensity = cry.get('intensity', 'medium')
        
        # Base severity by duration
        if duration < 10:
            base_severity = 'low'
        elif duration < 20:
            base_severity = 'medium'
        elif duration < 40:
            base_severity = 'high'
        else:
            base_severity = 'critical'
        
        # Adjust by intensity
        intensity_adjustment = {
            'low': 0,
            'medium': 1,
            'high': 2
        }
        
        severity_levels = ['low', 'medium', 'high', 'critical']
        current_index = severity_levels.index(base_severity)
        adjusted_index = min(current_index + intensity_adjustment.get(intensity, 1), 
                           len(severity_levels) - 1)
        
        return severity_levels[adjusted_index]
    
    def _detect_prolonged_silence(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect prolonged periods of silence that might indicate neglect
        זיהוי תקופות שקט ממושכות שעלולות להעיד על הזנחה
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            List of prolonged silence periods
        """
        silence_periods = []
        threshold = self.neglect_patterns['prolonged_silence']['energy_threshold']
        min_duration = self.neglect_patterns['prolonged_silence']['duration_threshold']
        
        # Analyze in 10-second windows
        window_length = 10.0
        window_samples = int(window_length * sr)
        hop_samples = window_samples // 2
        
        current_silence_start = None
        
        for start_sample in range(0, len(audio) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            window = audio[start_sample:end_sample]
            
            # Calculate energy for this window
            rms_energy = librosa.feature.rms(y=window)[0]
            mean_energy = np.mean(rms_energy)
            
            window_start_time = start_sample / sr
            window_end_time = end_sample / sr
            
            if mean_energy < threshold:
                # This window is silent
                if current_silence_start is None:
                    current_silence_start = window_start_time
            else:
                # This window has activity
                if current_silence_start is not None:
                    # End of silence period
                    silence_duration = window_start_time - current_silence_start
                    
                    if silence_duration >= min_duration:
                        silence_periods.append({
                            'start_time': current_silence_start,
                            'end_time': window_start_time,
                            'duration': silence_duration,
                            'severity': self._assess_silence_severity(silence_duration)
                        })
                    
                    current_silence_start = None
        
        # Handle case where audio ends during silence
        if current_silence_start is not None:
            silence_duration = len(audio) / sr - current_silence_start
            if silence_duration >= min_duration:
                silence_periods.append({
                    'start_time': current_silence_start,
                    'end_time': len(audio) / sr,
                    'duration': silence_duration,
                    'severity': self._assess_silence_severity(silence_duration)
                })
        
        return silence_periods
    
    def _assess_silence_severity(self, duration: float) -> str:
        """
        Assess severity of silence period
        הערכת חומרה של תקופת שקט
        
        Args:
            duration: Duration of silence in seconds
            
        Returns:
            Severity level
        """
        if duration < 180:  # Less than 3 minutes
            return 'low'
        elif duration < 300:  # Less than 5 minutes
            return 'medium'
        elif duration < 600:  # Less than 10 minutes
            return 'high'
        else:
            return 'critical'
    
    def _detect_lack_of_interaction(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect periods with lack of adult-child interaction
        זיהוי תקופות עם חוסר אינטראקציה בין מבוגרים לילדים
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            List of periods with lack of interaction
        """
        lack_of_interaction_periods = []
        interaction_window = self.neglect_patterns['lack_of_interaction']['interaction_window']
        adult_speech_threshold = self.neglect_patterns['lack_of_interaction']['adult_speech_threshold']
        
        # Analyze in 1-hour windows
        window_samples = int(interaction_window * sr)
        
        for start_sample in range(0, len(audio) - window_samples, window_samples):
            end_sample = start_sample + window_samples
            window = audio[start_sample:end_sample]
            
            window_start_time = start_sample / sr
            window_end_time = end_sample / sr
            
            # Calculate adult speech ratio in this window
            adult_speech_ratio = self._calculate_adult_speech_ratio(window, sr)
            
            if adult_speech_ratio < adult_speech_threshold:
                lack_of_interaction_periods.append({
                    'start_time': window_start_time,
                    'end_time': window_end_time,
                    'duration': interaction_window,
                    'adult_speech_ratio': adult_speech_ratio,
                    'severity': self._assess_interaction_severity(adult_speech_ratio)
                })
        
        return lack_of_interaction_periods
    
    def _calculate_adult_speech_ratio(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate ratio of adult speech in audio segment
        חישוב יחס דיבור מבוגרים בקטע אודיו
        
        Args:
            audio: Audio segment
            sr: Sample rate
            
        Returns:
            Ratio of adult speech (0-1)
        """
        # Segment into smaller chunks for analysis
        chunk_length = 2.0  # 2-second chunks
        chunk_samples = int(chunk_length * sr)
        hop_samples = chunk_samples // 2
        
        adult_speech_chunks = 0
        total_chunks = 0
        
        for start_sample in range(0, len(audio) - chunk_samples, hop_samples):
            end_sample = start_sample + chunk_samples
            chunk = audio[start_sample:end_sample]
            
            total_chunks += 1
            
            # Analyze chunk for adult speech
            if self._is_adult_speech_chunk(chunk, sr):
                adult_speech_chunks += 1
        
        return adult_speech_chunks / total_chunks if total_chunks > 0 else 0
    
    def _is_adult_speech_chunk(self, chunk: np.ndarray, sr: int) -> bool:
        """
        Determine if chunk contains adult speech
        קביעה אם חתיכה מכילה דיבור מבוגרים
        
        Args:
            chunk: Audio chunk
            sr: Sample rate
            
        Returns:
            True if chunk contains adult speech
        """
        # Check energy level
        rms_energy = librosa.feature.rms(y=chunk)[0]
        mean_energy = np.mean(rms_energy)
        
        if mean_energy < 0.05:
            return False
        
        # Check frequency characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
        mean_frequency = np.mean(spectral_centroid)
        
        # Adult speech typically in lower frequency range
        if 80 < mean_frequency < 400:
            # Check for speech characteristics
            zcr = librosa.feature.zero_crossing_rate(chunk)[0]
            mean_zcr = np.mean(zcr)
            
            if 0.01 < mean_zcr < 0.1:
                return True
        
        return False
    
    def _assess_interaction_severity(self, adult_speech_ratio: float) -> str:
        """
        Assess severity of lack of interaction
        הערכת חומרה של חוסר אינטראקציה
        
        Args:
            adult_speech_ratio: Ratio of adult speech
            
        Returns:
            Severity level
        """
        if adult_speech_ratio > 0.05:
            return 'low'
        elif adult_speech_ratio > 0.02:
            return 'medium'
        elif adult_speech_ratio > 0.005:
            return 'high'
        else:
            return 'critical'
    
    def _detect_ignored_distress(self, audio: np.ndarray, sr: int, 
                                violence_segments: List[Dict] = None) -> List[Dict]:
        """
        Detect episodes where distress was ignored
        זיהוי אירועים שבהם מצוקה לא זכתה לתשומת לב
        
        Args:
            audio: Audio data
            sr: Sample rate
            violence_segments: Previously detected violence segments
            
        Returns:
            List of ignored distress episodes
        """
        ignored_distress_episodes = []
        
        if not violence_segments:
            return ignored_distress_episodes
        
        for violence in violence_segments:
            # Look for response after violence incident
            response_window = self.neglect_thresholds['response_window']
            response_start = violence['end_time']
            response_end = min(response_start + response_window, len(audio) / sr)
            
            # Extract response period
            start_sample = int(response_start * sr)
            end_sample = int(response_end * sr)
            response_audio = audio[start_sample:end_sample]
            
            # Check for appropriate response
            has_appropriate_response = self._detect_appropriate_response(response_audio, sr, violence)
            
            if not has_appropriate_response:
                ignored_distress_episodes.append({
                    'violence_start_time': violence['start_time'],
                    'violence_end_time': violence['end_time'],
                    'violence_type': violence.get('violence_types', []),
                    'violence_severity': violence.get('severity', 'unknown'),
                    'response_window_start': response_start,
                    'response_window_end': response_end,
                    'severity': self._assess_distress_neglect_severity(violence)
                })
        
        return ignored_distress_episodes
    
    def _detect_appropriate_response(self, response_audio: np.ndarray, sr: int, 
                                   violence: Dict) -> bool:
        """
        Detect if there was an appropriate response to violence
        זיהוי אם הייתה תגובה מתאימה לאלימות
        
        Args:
            response_audio: Audio segment to analyze
            sr: Sample rate
            violence: Violence segment information
            
        Returns:
            True if appropriate response detected
        """
        if len(response_audio) < sr * 0.5:
            return False
        
        # Check for adult speech
        has_adult_speech = self._detect_staff_response(response_audio, sr)
        
        if not has_adult_speech:
            return False
        
        # For high-severity violence, check for immediate response
        violence_severity = violence.get('severity', 'low')
        if violence_severity in ['high', 'critical']:
            # Check if response was immediate (within first 30 seconds)
            response_duration = len(response_audio) / sr
            if response_duration < 30:
                return True
            else:
                # Check if there was immediate response in first 30 seconds
                immediate_response = response_audio[:int(30 * sr)]
                return self._detect_staff_response(immediate_response, sr)
        
        return True
    
    def _assess_distress_neglect_severity(self, violence: Dict) -> str:
        """
        Assess severity of neglecting distress
        הערכת חומרה של הזנחת מצוקה
        
        Args:
            violence: Violence segment information
            
        Returns:
            Severity level
        """
        violence_severity = violence.get('severity', 'low')
        violence_types = violence.get('violence_types', [])
        
        # Base severity on violence severity
        base_severity = violence_severity
        
        # Increase severity if multiple types of violence
        if len(violence_types) > 1:
            severity_levels = ['low', 'medium', 'high', 'critical']
            current_index = severity_levels.index(base_severity)
            adjusted_index = min(current_index + 1, len(severity_levels) - 1)
            return severity_levels[adjusted_index]
        
        return base_severity
    
    def _calculate_neglect_score(self, neglect_analysis: Dict) -> float:
        """
        Calculate overall neglect score
        חישוב ציון הזנחה כללי
        
        Args:
            neglect_analysis: Complete neglect analysis results
            
        Returns:
            Neglect score between 0 and 1
        """
        score = 0.0
        max_score = 0.0
        
        # Unanswered cries (weight: 0.4)
        unanswered_cries = neglect_analysis.get('unanswered_cries', [])
        if unanswered_cries:
            cry_scores = []
            for cry in unanswered_cries:
                severity_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
                cry_scores.append(severity_scores.get(cry.get('neglect_severity', 'low'), 0.2))
            
            score += np.mean(cry_scores) * 0.4
        max_score += 0.4
        
        # Prolonged silence (weight: 0.2)
        silence_periods = neglect_analysis.get('prolonged_silence_periods', [])
        if silence_periods:
            silence_scores = []
            for silence in silence_periods:
                severity_scores = {'low': 0.2, 'medium': 0.4, 'high': 0.6, 'critical': 0.8}
                silence_scores.append(severity_scores.get(silence.get('severity', 'low'), 0.2))
            
            score += np.mean(silence_scores) * 0.2
        max_score += 0.2
        
        # Lack of interaction (weight: 0.2)
        interaction_periods = neglect_analysis.get('lack_of_interaction_periods', [])
        if interaction_periods:
            interaction_scores = []
            for period in interaction_periods:
                severity_scores = {'low': 0.2, 'medium': 0.4, 'high': 0.6, 'critical': 0.8}
                interaction_scores.append(severity_scores.get(period.get('severity', 'low'), 0.2))
            
            score += np.mean(interaction_scores) * 0.2
        max_score += 0.2
        
        # Ignored distress (weight: 0.2)
        ignored_distress = neglect_analysis.get('ignored_distress_episodes', [])
        if ignored_distress:
            distress_scores = []
            for episode in ignored_distress:
                severity_scores = {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 1.0}
                distress_scores.append(severity_scores.get(episode.get('severity', 'low'), 0.3))
            
            score += np.mean(distress_scores) * 0.2
        max_score += 0.2
        
        return score / max_score if max_score > 0 else 0.0
    
    def _determine_neglect_severity(self, neglect_score: float) -> str:
        """
        Determine overall neglect severity based on score
        קביעת חומרת הזנחה כללית בהתבסס על ציון
        
        Args:
            neglect_score: Overall neglect score (0-1)
            
        Returns:
            Severity level
        """
        if neglect_score < 0.2:
            return 'none'
        elif neglect_score < 0.4:
            return 'low'
        elif neglect_score < 0.6:
            return 'medium'
        elif neglect_score < 0.8:
            return 'high'
        else:
            return 'critical'

if __name__ == "__main__":
    # Test the neglect detector
    detector = NeglectDetector()
    print("Neglect Detector initialized successfully")
    print("Ready to detect neglect patterns in kindergarten recordings...")
