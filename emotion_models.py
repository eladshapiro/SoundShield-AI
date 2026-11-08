"""Neural emotion models: categorical (HuBERT) + dimensional (arousal / dominance / valence).

``EmotionAnalyzer.timeline`` turns a recording into fixed windows and scores
each window with

* HuBERT-large SUPERB-ER  — P(anger), P(sad), P(happy), P(neutral)   (English-centric)
* audeering wav2vec2 MSP-Dim — arousal, dominance, valence in [0, 1]   (language-agnostic)

High arousal + high dominance in adult speech is the language-independent
signature of aggression, which matters for Hebrew kindergartens where an
English categorical model is unreliable.
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
HUBERT_LABELS = ('ang', 'hap', 'neu', 'sad')


@dataclass
class EmotionTimeline:
    starts: np.ndarray             # window start times (s)
    window: float
    duration: float
    arousal: np.ndarray            # [n] or empty when the dimensional model is unavailable
    dominance: np.ndarray
    valence: np.ndarray
    categorical: Dict[str, np.ndarray] = field(default_factory=dict)   # label -> [n]
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)    # 'w2v2' [n,1024], 'hubert' [n,256]
    heads: Dict[str, np.ndarray] = field(default_factory=dict)         # learned head -> P per window

    @property
    def has_dimensional(self) -> bool:
        return len(self.arousal) == len(self.starts) and len(self.starts) > 0

    @property
    def has_categorical(self) -> bool:
        return bool(self.categorical)

    def _mask(self, t0: float, t1: float) -> np.ndarray:
        return (self.starts + self.window > t0) & (self.starts < t1)

    def window_scores(self, name: str) -> np.ndarray:
        if name in ('arousal', 'dominance', 'valence'):
            return getattr(self, name)
        if name.startswith('head_'):
            return self.heads.get(name[5:], np.zeros(len(self.starts), dtype=np.float32))
        return self.categorical.get(name, np.zeros(len(self.starts), dtype=np.float32))

    def score(self, name: str, t0: float = 0.0, t1: Optional[float] = None, reduce: str = 'max') -> float:
        if t1 is None:
            t1 = self.duration
        m = self._mask(t0, t1)
        s = self.window_scores(name)
        if not m.any() or len(s) != len(m):
            return 0.0
        v = s[m]
        return float(v.max() if reduce == 'max' else v.mean())

    def to_dict(self) -> Dict:
        d = {'window': self.window, 'starts': [round(float(s), 3) for s in self.starts]}
        if self.has_dimensional:
            for k in ('arousal', 'dominance', 'valence'):
                d[k] = [round(float(v), 4) for v in getattr(self, k)]
        for k, v in self.categorical.items():
            d[f'p_{k}'] = [round(float(x), 4) for x in v]
        for k, v in self.heads.items():
            d[f'head_{k}'] = [round(float(x), 4) for x in v]
        return d


class DimensionalEmotionModel:
    """audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (arousal, dominance, valence)."""

    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.advanced.dim_emotion_model
        self._device_pref = device or 'auto'
        self.model = None
        self.processor = None
        self.device = 'cpu'
        self.loaded = False
        self.load_error: Optional[str] = None

    def load(self) -> bool:
        if self.loaded or self.load_error:
            return self.loaded
        with self._lock:
            if self.loaded:
                return True
            try:
                import torch
                import torch.nn as nn
                from transformers import Wav2Vec2Processor
                from transformers.models.wav2vec2.modeling_wav2vec2 import (
                    Wav2Vec2Model, Wav2Vec2PreTrainedModel)

                class RegressionHead(nn.Module):
                    def __init__(self, cfg):
                        super().__init__()
                        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
                        self.dropout = nn.Dropout(cfg.final_dropout)
                        self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

                    def forward(self, features):
                        x = self.dropout(features)
                        x = torch.tanh(self.dense(x))
                        x = self.dropout(x)
                        return self.out_proj(x)

                class EmotionModel(Wav2Vec2PreTrainedModel):
                    def __init__(self, cfg):
                        super().__init__(cfg)
                        self.config = cfg
                        self.wav2vec2 = Wav2Vec2Model(cfg)
                        self.classifier = RegressionHead(cfg)
                        self.post_init()

                    def forward(self, input_values):
                        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                        return self.classifier(hidden), hidden

                t0 = time.time()
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                model = EmotionModel.from_pretrained(self.model_name).eval()
                self.device = ('cuda' if torch.cuda.is_available() else 'cpu') \
                    if self._device_pref == 'auto' else self._device_pref
                self.model = model.to(self.device)
                self.loaded = True
                logger.info('DimensionalEmotionModel loaded %s on %s in %.1fs',
                            self.model_name, self.device, time.time() - t0)
            except Exception as e:
                self.load_error = f'{type(e).__name__}: {e}'
                logger.warning('Dimensional emotion model unavailable: %s', self.load_error)
        return self.loaded

    def to_cpu(self) -> None:
        if self.model is not None and self.device != 'cpu':
            self.model = self.model.to('cpu')
            self.device = 'cpu'

    def predict(self, chunks: Sequence[np.ndarray], return_hidden: bool = False):
        """[n, 3] array of (arousal, dominance, valence) for equal-length 16 kHz chunks.

        With ``return_hidden`` also the mean-pooled wav2vec2 hidden state [n, 1024].
        """
        import torch
        x = self.processor(list(chunks), sampling_rate=TARGET_SR, return_tensors='pt',
                           padding=True)['input_values'].to(self.device)
        with torch.no_grad():
            out, hidden = self.model(x)
        dims = out.float().cpu().numpy()
        if return_hidden:
            return dims, hidden.float().cpu().numpy()
        return dims


class CategoricalEmotionModel:
    """HuBERT SUPERB-ER via the transformers audio-classification pipeline."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.advanced.hubert_model
        self.pipe = None
        self.loaded = False
        self.load_error: Optional[str] = None
        self._tls = threading.local()      # per-thread capture of the pooled projector output
        self._hooked = False

    def _ensure_hook(self) -> None:
        """Keep the classifier's input (the pooled 256-d projector output) when the thread asks."""
        if self._hooked:
            return
        tls = self._tls

        def hook(module, inputs, output):
            if getattr(tls, 'want', False):
                tls.captured.append(inputs[0].detach().float().cpu().numpy())

        self.pipe.model.classifier.register_forward_hook(hook)
        self._hooked = True

    def load(self) -> bool:
        if self.loaded or self.load_error:
            return self.loaded
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline('audio-classification', model=self.model_name, device=device)
            self.loaded = True
        except Exception as e:
            self.load_error = f'{type(e).__name__}: {e}'
            logger.warning('HuBERT emotion model unavailable: %s', self.load_error)
        return self.loaded

    def to_cpu(self) -> None:
        import torch
        device = getattr(self.pipe, 'device', None) if self.pipe is not None else None
        if device is not None and torch.device(device).type != 'cpu':
            self.pipe.model.to('cpu')
            self.pipe.device = torch.device('cpu')

    def predict(self, chunks: Sequence[np.ndarray], return_hidden: bool = False):
        """Class probabilities per chunk; with ``return_hidden`` also the pooled embeddings [n, 256]."""
        if return_hidden:
            self._ensure_hook()
        self._tls.want = bool(return_hidden)
        self._tls.captured = []
        try:
            results = self.pipe([np.asarray(c, dtype=np.float32) for c in chunks], batch_size=8, top_k=None)
            hidden = np.concatenate(self._tls.captured) if return_hidden and self._tls.captured else None
        finally:
            self._tls.want = False
            self._tls.captured = []
        if chunks and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]
        probs = [{r['label'].lower(): float(r['score']) for r in res} for res in results]
        if return_hidden:
            return probs, hidden
        return probs


class EmotionAnalyzer:
    """Windowed emotion scoring combining the dimensional and categorical models."""

    def __init__(self, window: Optional[float] = None, hop: Optional[float] = None,
                 use_dimensional: bool = True, use_categorical: bool = True,
                 categorical: Optional[CategoricalEmotionModel] = None):
        cfg = config.advanced
        self.window = float(window or cfg.emotion_window_seconds)
        self.hop = float(hop or cfg.emotion_hop_seconds)
        self.dim = DimensionalEmotionModel() if use_dimensional else None
        self.cat = (categorical or CategoricalEmotionModel()) if use_categorical else None

    @property
    def available(self) -> bool:
        return bool((self.dim and self.dim.load()) or (self.cat and self.cat.load()))

    def _windows(self, n: int) -> np.ndarray:
        win = int(self.window * TARGET_SR)
        hop = int(self.hop * TARGET_SR)
        if n <= win:
            return np.array([0.0])
        starts = np.arange(0, n - win + 1, hop)
        if starts[-1] + win < n:
            starts = np.append(starts, n - win)
        return starts / TARGET_SR

    def timeline(self, audio: np.ndarray, sr: int, batch_size: int = 16,
                 embeddings: bool = False) -> Optional[EmotionTimeline]:
        """Score fixed windows; ``embeddings=True`` also keeps the models' pooled embeddings."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        duration = len(audio) / TARGET_SR
        starts = self._windows(len(audio))
        win = int(self.window * TARGET_SR)
        chunks = []
        for s in starts:
            i = int(round(s * TARGET_SR))
            c = audio[i:i + win]
            if len(c) < int(0.5 * TARGET_SR):
                c = np.pad(c, (0, int(0.5 * TARGET_SR) - len(c)))
            chunks.append(c)
        use_dim = self.dim is not None and self.dim.load()
        use_cat = self.cat is not None and self.cat.load()
        if not use_dim and not use_cat:
            return None
        from cuda_compat import is_cuda_oom
        try:
            scored = self._score(chunks, batch_size, embeddings, use_dim, use_cat)
        except RuntimeError as e:
            if not is_cuda_oom(e):
                raise
            # not even one window fits on the GPU (another program holds it): finish on the CPU
            # rather than letting the analysis fall back to the heuristics
            self.to_cpu()
            scored = self._score(chunks, batch_size, embeddings, use_dim, use_cat)
        arousal, dominance, valence, categorical, emb = scored
        return EmotionTimeline(starts=np.asarray(starts, dtype=np.float64), window=self.window,
                               duration=duration, arousal=arousal, dominance=dominance,
                               valence=valence, categorical=categorical, embeddings=emb)

    def to_cpu(self) -> None:
        """Move both models to the CPU for the rest of the process (last resort on a full GPU)."""
        logger.warning('GPU memory exhausted even for one window: emotion models move to the CPU')
        if self.dim is not None:
            self.dim.to_cpu()
        if self.cat is not None:
            self.cat.to_cpu()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _score(self, chunks: List[np.ndarray], batch_size: int, embeddings: bool,
               use_dim: bool, use_cat: bool):
        # equalise lengths inside a batch (last window may be shorter on very short clips)
        n = len(chunks)
        arousal = dominance = valence = np.zeros(0, dtype=np.float32)
        categorical: Dict[str, np.ndarray] = {}
        from cuda_compat import halve_on_oom      # a full GPU shrinks the batch instead of failing
        emb: Dict[str, np.ndarray] = {}
        if use_dim:
            dims = np.zeros((n, 3), dtype=np.float32)
            hidden = []
            for b in range(0, n, batch_size):
                batch = chunks[b:b + batch_size]
                L = max(len(c) for c in batch)
                batch = [np.pad(c, (0, L - len(c))) for c in batch]
                if embeddings:
                    out, h = halve_on_oom(lambda x: self.dim.predict(x, return_hidden=True), batch)
                    hidden.append(h)
                else:
                    out = halve_on_oom(self.dim.predict, batch)
                dims[b:b + len(batch)] = out
            arousal, dominance, valence = dims[:, 0], dims[:, 1], dims[:, 2]
            if embeddings:
                emb['w2v2'] = np.concatenate(hidden).astype(np.float32)
        if use_cat:
            cat = {k: np.zeros(n, dtype=np.float32) for k in HUBERT_LABELS}
            hidden = []
            for b in range(0, n, batch_size):
                res = halve_on_oom(lambda x: self.cat.predict(x, return_hidden=embeddings),
                                   chunks[b:b + batch_size])
                if embeddings:
                    res, h = res
                    if h is not None:
                        hidden.append(h)
                for j, scores in enumerate(res):
                    for k in HUBERT_LABELS:
                        cat[k][b + j] = scores.get(k, 0.0)
            categorical = cat
            if embeddings and hidden and sum(len(h) for h in hidden) == n:
                emb['hubert'] = np.concatenate(hidden).astype(np.float32)
        return arousal, dominance, valence, categorical, emb


# ---------------------------------------------------------------------------
# Decision rules shared by the production detectors and the evaluation suite
# ---------------------------------------------------------------------------
def anger_rule(p_anger: float, arousal: Optional[float], dominance: Optional[float],
               valence: Optional[float]) -> Tuple[bool, float, str]:
    """Concerning-emotion decision for one window.

    Returns (flag, confidence, emotion) where emotion is 'anger' when the
    categorical model agrees and 'aggression' when only the dimensional
    model fires.  Rule (config.advanced): arousal >= A and dominance >= D and
    (valence <= V or P(anger) >= T).  Without the dimensional model, falls back
    to P(anger) >= T alone.
    """
    cfg = config.advanced
    if arousal is None or dominance is None:
        flag = p_anger >= cfg.anger_prob_threshold
        return flag, float(p_anger), 'anger'
    dims_ok = arousal >= cfg.arousal_threshold and dominance >= cfg.dominance_threshold
    low_valence = valence is not None and valence <= cfg.valence_max
    hubert_ok = p_anger >= cfg.anger_prob_threshold
    if dims_ok and (low_valence or hubert_ok):
        if hubert_ok:
            return True, float(max(p_anger, min(arousal, dominance))), 'anger'
        return True, float(min(arousal, dominance)), 'aggression'
    return False, 0.0, ''


def aggressive_rule(arousal: float, dominance: float, valence: Optional[float]) -> Tuple[bool, float]:
    """Aggressive-tone decision for the violence detector (config.violence)."""
    cfg = config.violence
    if arousal >= cfg.aggressive_arousal and dominance >= cfg.aggressive_dominance \
            and (valence is None or valence <= cfg.aggressive_valence_max):
        return True, float(min(arousal, dominance))
    return False, 0.0


_shared: Optional[EmotionAnalyzer] = None


def get_emotion_analyzer() -> Optional[EmotionAnalyzer]:
    global _shared
    if not config.advanced.use_advanced:
        return None
    if _shared is None:
        _shared = EmotionAnalyzer()
    return _shared
