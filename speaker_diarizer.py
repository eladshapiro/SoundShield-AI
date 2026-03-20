"""
Speaker Diarization Module for SoundShield-AI
"""
import logging
import numpy as np
import librosa
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import pyannote
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("pyannote.audio not available - speaker diarization disabled")


class SpeakerDiarizer:
    """Diarize audio into speaker segments and classify adult vs child."""

    def __init__(self, auth_token: Optional[str] = None):
        """Initialize speaker diarizer.

        Args:
            auth_token: HuggingFace auth token for pyannote models (optional)
        """
        self.pipeline = None
        self.available = False
        self._load_pipeline(auth_token)

    def _load_pipeline(self, auth_token):
        """Load pyannote pipeline. Falls back to pitch-based if unavailable."""
        if not PYANNOTE_AVAILABLE:
            logger.info("Using pitch-based speaker classification (pyannote not available)")
            return
        try:
            kwargs = {}
            if auth_token:
                kwargs['use_auth_token'] = auth_token
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", **kwargs
            )
            self.available = True
            logger.info("pyannote speaker diarization pipeline loaded")
        except Exception as e:
            logger.warning(f"Could not load pyannote pipeline: {e}")
            logger.info("Falling back to pitch-based speaker classification")

    def get_speaker_segments(self, audio_path: str) -> List[Dict]:
        """Get speaker segments with adult/child classification.

        Args:
            audio_path: Path to audio file

        Returns:
            List of dicts with keys: speaker_id, start_time, end_time, is_adult, avg_f0
        """
        if self.available and self.pipeline:
            return self._diarize_with_pyannote(audio_path)
        else:
            return self._diarize_with_pitch(audio_path)

    def _diarize_with_pyannote(self, audio_path: str) -> List[Dict]:
        """Use pyannote pipeline for diarization, then classify by pitch."""
        try:
            diarization = self.pipeline(audio_path)
            audio, sr = librosa.load(audio_path, sr=16000)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end

                # Extract segment audio for pitch analysis
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]

                avg_f0 = self._estimate_f0(segment_audio, sr)
                is_adult = avg_f0 < 200.0 if avg_f0 > 0 else True  # Default to adult if no pitch detected

                segments.append({
                    'speaker_id': speaker,
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'is_adult': is_adult,
                    'avg_f0': float(avg_f0)
                })

            logger.info(f"Diarized {len(segments)} speaker segments via pyannote")
            return segments

        except Exception as e:
            logger.error(f"pyannote diarization failed: {e}, falling back to pitch-based")
            return self._diarize_with_pitch(audio_path)

    def _diarize_with_pitch(self, audio_path: str) -> List[Dict]:
        """Fallback: segment by energy/silence and classify by F0 pitch.

        Uses simple energy-based segmentation with pitch analysis.
        < 200Hz fundamental frequency = likely adult
        > 200Hz = likely child
        """
        try:
            audio, sr = librosa.load(audio_path, sr=16000)

            # Segment by voice activity (energy-based)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

            # Threshold for voice activity
            threshold = np.mean(rms) * 0.5
            is_voice = rms > threshold

            # Convert to segments
            segments = []
            times = librosa.frames_to_time(np.arange(len(is_voice)), sr=sr, hop_length=hop_length)

            in_segment = False
            seg_start = 0.0
            speaker_count = 0

            for i, voiced in enumerate(is_voice):
                if voiced and not in_segment:
                    seg_start = times[i]
                    in_segment = True
                elif not voiced and in_segment:
                    seg_end = times[i]
                    if seg_end - seg_start > 0.3:  # Min 300ms segments
                        start_sample = int(seg_start * sr)
                        end_sample = int(seg_end * sr)
                        segment_audio = audio[start_sample:end_sample]

                        avg_f0 = self._estimate_f0(segment_audio, sr)
                        is_adult = avg_f0 < 200.0 if avg_f0 > 0 else True

                        segments.append({
                            'speaker_id': f"SPEAKER_{speaker_count % 5}",
                            'start_time': float(seg_start),
                            'end_time': float(seg_end),
                            'is_adult': is_adult,
                            'avg_f0': float(avg_f0)
                        })
                        speaker_count += 1
                    in_segment = False

            # Handle segment at end of audio
            if in_segment:
                seg_end = times[-1] if len(times) > 0 else len(audio) / sr
                if seg_end - seg_start > 0.3:
                    start_sample = int(seg_start * sr)
                    end_sample = min(int(seg_end * sr), len(audio))
                    segment_audio = audio[start_sample:end_sample]
                    avg_f0 = self._estimate_f0(segment_audio, sr)
                    is_adult = avg_f0 < 200.0 if avg_f0 > 0 else True
                    segments.append({
                        'speaker_id': f"SPEAKER_{speaker_count % 5}",
                        'start_time': float(seg_start),
                        'end_time': float(seg_end),
                        'is_adult': is_adult,
                        'avg_f0': float(avg_f0)
                    })

            logger.info(f"Diarized {len(segments)} speaker segments via pitch analysis")
            return segments

        except Exception as e:
            logger.error(f"Pitch-based diarization failed: {e}")
            return []

    def _estimate_f0(self, audio: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency using pyin.

        Returns average F0 in Hz, or 0.0 if unvoiced.
        """
        try:
            if len(audio) < sr * 0.1:  # Need at least 100ms
                return 0.0

            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=50, fmax=500, sr=sr
            )

            # Filter to voiced frames only
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

            if len(voiced_f0) == 0:
                return 0.0

            return float(np.median(voiced_f0))

        except Exception:
            return 0.0

    def get_summary(self, segments: List[Dict]) -> Dict:
        """Summarize diarization results.

        Returns:
            Dict with speaker_count, adult_count, child_count,
            total_adult_time, total_child_time
        """
        if not segments:
            return {
                'speaker_count': 0,
                'adult_count': 0,
                'child_count': 0,
                'total_adult_time': 0.0,
                'total_child_time': 0.0,
                'speakers': {}
            }

        speakers = {}
        for seg in segments:
            sid = seg['speaker_id']
            if sid not in speakers:
                speakers[sid] = {
                    'is_adult': seg['is_adult'],
                    'total_time': 0.0,
                    'segment_count': 0,
                    'avg_f0_values': []
                }
            speakers[sid]['total_time'] += seg['end_time'] - seg['start_time']
            speakers[sid]['segment_count'] += 1
            if seg['avg_f0'] > 0:
                speakers[sid]['avg_f0_values'].append(seg['avg_f0'])

        # Finalize speaker info
        for sid, info in speakers.items():
            if info['avg_f0_values']:
                info['avg_f0'] = float(np.mean(info['avg_f0_values']))
            else:
                info['avg_f0'] = 0.0
            del info['avg_f0_values']

        adult_speakers = {k: v for k, v in speakers.items() if v['is_adult']}
        child_speakers = {k: v for k, v in speakers.items() if not v['is_adult']}

        return {
            'speaker_count': len(speakers),
            'adult_count': len(adult_speakers),
            'child_count': len(child_speakers),
            'total_adult_time': sum(s['total_time'] for s in adult_speakers.values()),
            'total_child_time': sum(s['total_time'] for s in child_speakers.values()),
            'speakers': speakers
        }
