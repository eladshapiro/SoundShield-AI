"""Per-clip prediction runners that exercise the production detector code paths.

Heuristic runners are process-safe (detectors are built lazily per worker).
Model runners hold a model and run in the main process (GPU if available).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

_DETECTORS: Dict[str, Any] = {}


def _get(name: str):
    if name not in _DETECTORS:
        if name == 'cry':
            from cry_detector import CryDetector
            _DETECTORS[name] = CryDetector()
        elif name == 'violence':
            from violence_detector import ViolenceDetector
            _DETECTORS[name] = ViolenceDetector()
        elif name == 'emotion':
            from emotion_detector import EmotionDetector
            _DETECTORS[name] = EmotionDetector()
        elif name == 'neglect':
            from neglect_detector import NeglectDetector
            _DETECTORS[name] = NeglectDetector()
        elif name == 'audio':
            from audio_analyzer import AudioAnalyzer
            _DETECTORS[name] = AudioAnalyzer()
        else:
            raise KeyError(name)
    return _DETECTORS[name]


def load_audio(path: str, sr: int) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


# ---------------------------------------------------------------------------
# Heuristic detectors (production code paths)
# ---------------------------------------------------------------------------
def run_heuristics(path: str) -> Dict[str, Any]:
    """Run cry / violence / emotion / speech heuristics exactly as the pipeline does."""
    from config import config
    sr = config.audio.sample_rate
    audio = load_audio(path, sr)
    duration = len(audio) / sr
    out: Dict[str, Any] = {'path': path, 'duration': duration}

    # Cry detector -------------------------------------------------------
    cry = _get('cry')
    segs = cry.detect_cry_segments(audio, sr)
    out['cry_pred'] = int(bool(segs))
    out['cry_score'] = float(max((s['confidence'] for s in segs), default=0.0))
    out['cry_coverage'] = float(sum(s['duration'] for s in segs) / duration) if duration else 0.0

    # Violence detector --------------------------------------------------
    viol = _get('violence').detect_violence_segments(audio, sr)
    out['violence_pred'] = int(bool(viol))
    out['violence_score'] = float(max((s['confidence'] for s in viol), default=0.0))
    out['violence_types'] = sorted({t for s in viol for t in s['violence_types']})

    # Heuristic emotion on 5 s segments ----------------------------------
    analyzer = _get('audio')
    segments = analyzer.segment_audio(audio, config.audio.segment_length)
    emo = _get('emotion')
    seg_emotions = emo.analyze_segment_emotions(segments, sr) if segments else []
    concerning = emo.detect_concerning_emotions(seg_emotions)
    anger = [c for c in concerning if c['detected_emotion'] in ('anger', 'aggression')]
    out['anger_pred'] = int(bool(anger))
    out['anger_score'] = float(max((c['confidence'] for c in anger), default=0.0))
    out['concerning_any'] = int(bool(concerning))
    out['primary_emotions'] = [s['emotion_analysis']['primary_emotion'] for s in seg_emotions]

    # Speech / adult-response presence -----------------------------------
    win = sr  # 1 s windows as in CryDetector._analyze_response_segment
    resp_hits = 0
    n = 0
    for start in range(0, max(1, len(audio) - win + 1), win // 2):
        chunk = audio[start:start + win]
        if len(chunk) < win * 0.5:
            break
        n += 1
        if cry._is_response_segment(cry._calculate_response_features(chunk, sr)):
            resp_hits += 1
    out['speech_pred'] = int(resp_hits > 0)
    out['speech_score'] = resp_hits / n if n else 0.0
    neg = _get('neglect')
    win2 = 2 * sr
    adult_hits = 0
    n2 = 0
    for start in range(0, max(1, len(audio) - win2 + 1), sr):
        chunk = audio[start:start + win2]
        if len(chunk) < sr:
            break
        n2 += 1
        if neg._is_adult_speech_chunk(chunk, sr):
            adult_hits += 1
    out['adult_speech_pred'] = int(adult_hits > 0)
    out['adult_speech_score'] = adult_hits / n2 if n2 else 0.0
    return out


# ---------------------------------------------------------------------------
# HuBERT emotion (AdvancedAnalyzer.detect_concerning_emotions_advanced)
# ---------------------------------------------------------------------------
class HubertRunner:
    def __init__(self):
        from advanced_analyzer import AdvancedAnalyzer
        self.analyzer = AdvancedAnalyzer(use_whisper=False, use_transformer_emotion=True)
        self.analyzer.load_models()
        if not self.analyzer.hubert_loaded:
            raise RuntimeError('HuBERT failed to load')

    def __call__(self, path: str) -> Dict[str, Any]:
        res = self.analyzer.detect_concerning_emotions_advanced(path)
        anger = [r for r in res if r['detected_emotion'] in ('anger', 'aggression')]
        return {
            'path': path,
            'anger_pred': int(bool(anger)),
            'anger_score': float(max((r['confidence'] for r in anger), default=0.0)),
            'concerning_any': int(bool(res)),
        }


# ---------------------------------------------------------------------------
# Hebrew ASR (WER on FLEURS)
# ---------------------------------------------------------------------------
_HEB_NIKUD = re.compile(r'[֑-ׇ]')
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)


def normalize_hebrew(text: str) -> str:
    text = _HEB_NIKUD.sub('', text)
    text = _PUNCT.sub(' ', text)
    return ' '.join(text.split()).lower()


class WhisperRunner:
    """Transcribe with faster-whisper; ``model_name`` may be a HF repo or size."""

    def __init__(self, model_name: str, device: Optional[str] = None,
                 compute_type: Optional[str] = None, beam_size: int = 5):
        from cuda_compat import preload_cuda12_libs, cuda_available
        preload_cuda12_libs()
        from faster_whisper import WhisperModel
        if device is None:
            device = 'cuda' if cuda_available() else 'cpu'
        if compute_type is None:
            compute_type = 'float16' if device == 'cuda' else 'int8'
        self.name = model_name
        self.beam_size = beam_size
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def __call__(self, path: str, language: str = 'he') -> Dict[str, Any]:
        segs, _ = self.model.transcribe(path, language=language, beam_size=self.beam_size)
        text = ' '.join(s.text for s in segs)
        return {'path': path, 'hypothesis': text}


# ---------------------------------------------------------------------------
# AudioSet tagger (audio_tagger.AudioTagger) — cry / scream / speech
# ---------------------------------------------------------------------------
class TaggerRunner:
    GROUPS = ('cry', 'infant_cry', 'scream', 'speech', 'adult_speech', 'child_speech',
              'laughter', 'impact', 'silence', 'music')

    def __init__(self):
        from audio_tagger import AudioTagger
        from config import config
        self.cfg = config.tagger
        self.tagger = AudioTagger()
        if not self.tagger.load():
            raise RuntimeError(f'AudioTagger failed to load: {self.tagger.load_error}')

    def __call__(self, path: str) -> Dict[str, Any]:
        tl = self.tagger.tag_file(path)
        out: Dict[str, Any] = {'path': path, 'duration': tl.duration, 'n_windows': int(len(tl.starts))}
        for g in self.GROUPS:
            s = tl.group_scores(g)
            out[f'{g}_max'] = float(s.max()) if len(s) else 0.0
            out[f'{g}_mean'] = float(s.mean()) if len(s) else 0.0
        out['cry_score'] = out['cry_max']
        out['cry_pred'] = int(out['cry_max'] >= self.cfg.cry_threshold)
        out['violence_score'] = out['scream_max']
        out['violence_pred'] = int(out['scream_max'] >= self.cfg.scream_threshold)
        out['speech_score'] = out['speech_max']
        out['speech_pred'] = int(out['speech_max'] >= self.cfg.speech_threshold)
        out['top'] = tl.top_labels(0, tl.duration, k=3)
        return out


# ---------------------------------------------------------------------------
# Neural emotion (emotion_models.EmotionAnalyzer) — anger / aggression
# ---------------------------------------------------------------------------
class Emotion2Runner:
    def __init__(self):
        from emotion_models import EmotionAnalyzer
        from config import config
        self.cfg = config.advanced
        self.analyzer = EmotionAnalyzer()
        if not self.analyzer.available:
            raise RuntimeError('no emotion model available')

    def __call__(self, path: str) -> Dict[str, Any]:
        from emotion_models import anger_rule, aggressive_rule
        audio = load_audio(path, 16000)
        tl = self.analyzer.timeline(audio, 16000)
        out: Dict[str, Any] = {'path': path, 'duration': tl.duration, 'n_windows': int(len(tl.starts))}
        n = len(tl.starts)
        p_ang = tl.categorical.get('ang', np.zeros(n))
        out['windows'] = {'p_ang': [round(float(x), 4) for x in p_ang]}
        if tl.has_dimensional:
            for k in ('arousal', 'dominance', 'valence'):
                v = getattr(tl, k)
                out[f'{k}_max'] = float(v.max())
                out[f'{k}_mean'] = float(v.mean())
                out['windows'][k] = [round(float(x), 4) for x in v]
            out['aggr_max'] = float(np.minimum(tl.arousal, tl.dominance).max())
        for k, v in tl.categorical.items():
            out[f'p_{k}_max'] = float(v.max())
            out[f'p_{k}_mean'] = float(v.mean())
        # production rules evaluated per window (same functions the detectors use)
        anger_flags, anger_conf, aggr_flags, aggr_conf = [], [], [], []
        for i in range(n):
            if tl.has_dimensional:
                a, d, v = float(tl.arousal[i]), float(tl.dominance[i]), float(tl.valence[i])
            else:
                a = d = v = None
            f, c, _ = anger_rule(float(p_ang[i]), a, d, v)
            anger_flags.append(f); anger_conf.append(c)
            if tl.has_dimensional:
                f2, c2 = aggressive_rule(a, d, v)
                aggr_flags.append(f2); aggr_conf.append(c2)
        out['anger_pred'] = int(any(anger_flags))
        out['anger_score'] = float(max(anger_conf) if anger_conf else 0.0)
        out['violence_pred'] = int(any(aggr_flags))
        out['violence_score'] = float(max(aggr_conf) if aggr_conf else 0.0)
        return out
