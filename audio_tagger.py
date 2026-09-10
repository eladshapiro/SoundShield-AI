"""AudioSet event tagger — the ML-first signal for cry / scream / speech detection.

Wraps the Audio Spectrogram Transformer fine-tuned on AudioSet (527 classes)
and turns a recording into a window-level timeline of event probabilities.
Detectors consume grouped scores (``cry``, ``scream``, ``speech``, ...)
instead of hand-tuned spectral thresholds; the heuristic detectors remain the
fallback when the model is unavailable.

Usage
-----
    tagger = AudioTagger()                # loads model lazily on first call
    timeline = tagger.tag(audio16k, 16000)
    timeline.score('cry', t0=12.0, t1=15.0)   -> max cry probability in [12, 15)
    timeline.segments('cry', threshold=0.3)   -> [(start, end, peak_score), ...]
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)

TARGET_SR = 16000

# AudioSet label groups.  Names must match ``model.config.id2label`` exactly.
LABEL_GROUPS: Dict[str, Tuple[str, ...]] = {
    'cry': ('Baby cry, infant cry', 'Crying, sobbing', 'Whimper', 'Wail, moan'),
    'infant_cry': ('Baby cry, infant cry',),
    'scream': ('Screaming', 'Shout', 'Yell', 'Battle cry', 'Children shouting'),
    'speech': ('Speech', 'Male speech, man speaking', 'Female speech, woman speaking',
               'Conversation', 'Narration, monologue', 'Child speech, kid speaking', 'Chatter'),
    'adult_speech': ('Male speech, man speaking', 'Female speech, woman speaking',
                     'Narration, monologue', 'Conversation'),
    'child_speech': ('Child speech, kid speaking', 'Children playing', 'Children shouting',
                     'Child singing', 'Babbling'),
    'laughter': ('Laughter', 'Baby laughter', 'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle'),
    'impact': ('Slap, smack', 'Whack, thwack', 'Smash, crash', 'Breaking', 'Bang', 'Thump, thud', 'Glass'),
    'silence': ('Silence',),
    'music': ('Music', 'Singing', 'Music for children'),
}


@dataclass
class TagTimeline:
    """Window-level AudioSet probabilities for one recording."""
    starts: np.ndarray                 # window start times (s), shape [n]
    window: float                      # window length (s)
    probs: np.ndarray                  # sigmoid outputs, shape [n, n_labels] (float32)
    labels: List[str]                  # id -> name
    duration: float
    group_index: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        name_to_id = {n: i for i, n in enumerate(self.labels)}
        for g, names in LABEL_GROUPS.items():
            ids = [name_to_id[n] for n in names if n in name_to_id]
            self.group_index[g] = np.asarray(ids, dtype=int)
        self._derived: Dict[str, np.ndarray] = {}

    # -- queries ----------------------------------------------------------
    def group_scores(self, group: str) -> np.ndarray:
        """Per-window score for a group = max probability over its labels.

        ``speech_dominant`` is a derived group: the speech score where speech
        exceeds the cry score by ``config.tagger.speech_over_cry_margin`` and
        0 elsewhere.  A crying child's vocalisations partially activate the
        AudioSet speech labels, so staff-response logic must use this group
        rather than raw ``speech``.
        """
        if group == 'speech_dominant':
            if group not in self._derived:
                speech = self.group_scores('speech')
                cry = self.group_scores('cry')
                margin = config.tagger.speech_over_cry_margin
                self._derived[group] = np.where(speech > cry + margin, speech, 0.0).astype(np.float32)
            return self._derived[group]
        ids = self.group_index.get(group)
        if ids is None or len(ids) == 0 or len(self.probs) == 0:
            return np.zeros(len(self.starts), dtype=np.float32)
        return self.probs[:, ids].max(axis=1)

    def label_scores(self, label: str) -> np.ndarray:
        idx = self.labels.index(label)
        return self.probs[:, idx]

    def _window_mask(self, t0: float, t1: float) -> np.ndarray:
        ends = self.starts + self.window
        return (ends > t0) & (self.starts < t1)

    def score(self, group: str, t0: float = 0.0, t1: Optional[float] = None,
              reduce: str = 'max') -> float:
        """Reduce a group's window scores over [t0, t1)."""
        if t1 is None:
            t1 = self.duration
        m = self._window_mask(t0, t1)
        if not m.any():
            return 0.0
        s = self.group_scores(group)[m]
        return float(s.max() if reduce == 'max' else s.mean())

    def top_labels(self, t0: float, t1: float, k: int = 5) -> List[Tuple[str, float]]:
        m = self._window_mask(t0, t1)
        if not m.any():
            return []
        p = self.probs[m].max(axis=0)
        order = np.argsort(-p)[:k]
        return [(self.labels[int(i)], float(p[i])) for i in order]

    def segments(self, group: str, threshold: float, min_windows: int = 1,
                 merge_gap: float = 0.0) -> List[Tuple[float, float, float]]:
        """Contiguous runs of windows whose group score >= threshold.

        Returns (start, end, peak_score) tuples in seconds.
        """
        scores = self.group_scores(group)
        hits = scores >= threshold
        out: List[Tuple[float, float, float]] = []
        i = 0
        n = len(hits)
        while i < n:
            if not hits[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and hits[j + 1]:
                j += 1
            if j - i + 1 >= min_windows:
                start = float(self.starts[i])
                end = float(min(self.starts[j] + self.window, self.duration))
                peak = float(scores[i:j + 1].max())
                if out and start - out[-1][1] <= merge_gap:
                    ps, pe, pp = out[-1]
                    out[-1] = (ps, max(pe, end), max(pp, peak))
                else:
                    out.append((start, end, peak))
            i = j + 1
        return out

    def to_dict(self, groups: Sequence[str] = ('cry', 'infant_cry', 'scream', 'speech', 'speech_dominant',
                                               'child_speech', 'laughter', 'impact', 'silence')) -> Dict:
        return {
            'window': self.window,
            'hop': float(self.starts[1] - self.starts[0]) if len(self.starts) > 1 else self.window,
            'starts': [round(float(s), 3) for s in self.starts],
            'groups': {g: [round(float(v), 4) for v in self.group_scores(g)] for g in groups},
        }


class AudioTagger:
    """Lazy-loading AST tagger with batched GPU/CPU inference."""

    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, window: Optional[float] = None,
                 hop: Optional[float] = None, batch_size: Optional[int] = None,
                 device: Optional[str] = None):
        cfg = config.tagger
        self.model_name = model_name or cfg.model
        self.window = float(window or cfg.window_seconds)
        self.hop = float(hop or cfg.hop_seconds)
        self.batch_size = int(batch_size or cfg.batch_size)
        self._device_pref = device or cfg.device
        self.model = None
        self.extractor = None
        self.labels: List[str] = []
        self.device = 'cpu'
        self.loaded = False
        self.load_error: Optional[str] = None

    # -- loading ------------------------------------------------------------
    def load(self) -> bool:
        if self.loaded or self.load_error:
            return self.loaded
        with self._lock:
            if self.loaded:
                return True
            try:
                import torch
                from transformers import AutoFeatureExtractor, ASTForAudioClassification
                t0 = time.time()
                self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = ASTForAudioClassification.from_pretrained(self.model_name).eval()
                if self._device_pref == 'auto':
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    self.device = self._device_pref
                model = model.to(self.device)
                if self.device == 'cuda':
                    model = model.half()
                self.model = model
                self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
                self.loaded = True
                logger.info('AudioTagger loaded %s on %s in %.1fs', self.model_name, self.device, time.time() - t0)
            except Exception as e:  # model missing / no transformers / OOM
                self.load_error = f'{type(e).__name__}: {e}'
                logger.warning('AudioTagger unavailable (%s); heuristic detectors will be used', self.load_error)
        return self.loaded

    # -- inference ----------------------------------------------------------
    @staticmethod
    def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio

    def windows(self, n_samples: int) -> np.ndarray:
        win = int(self.window * TARGET_SR)
        hop = int(self.hop * TARGET_SR)
        if n_samples <= win:
            return np.array([0.0])
        starts = np.arange(0, n_samples - win + 1, hop)
        if starts[-1] + win < n_samples:                       # cover the tail
            starts = np.append(starts, n_samples - win)
        return starts / TARGET_SR

    def tag(self, audio: np.ndarray, sr: int) -> Optional[TagTimeline]:
        """Tag a whole recording. Returns None when the model is unavailable."""
        if not self.load():
            return None
        import torch
        audio = self._to_16k(audio, sr)
        duration = len(audio) / TARGET_SR
        starts = self.windows(len(audio))
        win = int(self.window * TARGET_SR)
        probs = np.zeros((len(starts), len(self.labels)), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for b in range(0, len(starts), self.batch_size):
                chunk_starts = starts[b:b + self.batch_size]
                chunks = []
                for s in chunk_starts:
                    i = int(round(s * TARGET_SR))
                    c = audio[i:i + win]
                    if len(c) < win:
                        c = np.pad(c, (0, win - len(c)))
                    chunks.append(c)
                feats = self._features(chunks)
                logits = self.model(feats).logits.float()
                probs[b:b + len(chunk_starts)] = torch.sigmoid(logits).cpu().numpy()
        logger.debug('AudioTagger: %d windows in %.2fs (%.1fs audio)', len(starts), time.time() - t0, duration)
        return TagTimeline(starts=np.asarray(starts, dtype=np.float64), window=self.window,
                           probs=probs, labels=self.labels, duration=duration)

    def _features(self, chunks: List[np.ndarray]):
        """AST input features for a batch of equal-length 16 kHz windows.

        On CUDA the Kaldi filterbank is computed on the GPU (same parameters
        as ``ASTFeatureExtractor``); on CPU the HF extractor is used.
        """
        import torch
        if self.device != 'cuda':
            feats = self.extractor(chunks, sampling_rate=TARGET_SR, return_tensors='pt')['input_values']
            return feats.to(self.device)
        import torchaudio.compliance.kaldi as ta_kaldi
        ex = self.extractor
        max_len = int(getattr(ex, 'max_length', 1024))
        wav = torch.from_numpy(np.stack(chunks)).to(self.device)
        out = torch.zeros((len(chunks), max_len, ex.num_mel_bins), device=self.device)
        for i in range(len(chunks)):
            fb = ta_kaldi.fbank(wav[i:i + 1], htk_compat=True, sample_frequency=TARGET_SR,
                                use_energy=False, window_type='hanning',
                                num_mel_bins=ex.num_mel_bins, dither=0.0, frame_shift=10)
            n = min(fb.shape[0], max_len)
            out[i, :n] = fb[:n]
        if getattr(ex, 'do_normalize', True):
            out = (out - ex.mean) / (ex.std * 2)
        return out.half()

    def tag_file(self, path: str) -> Optional[TagTimeline]:
        import librosa
        audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        return self.tag(audio, sr)


_shared: Optional[AudioTagger] = None


def get_tagger() -> Optional[AudioTagger]:
    """Process-wide shared tagger (None when disabled in config)."""
    global _shared
    if not config.tagger.enabled:
        return None
    if _shared is None:
        _shared = AudioTagger()
    return _shared
