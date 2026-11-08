"""Learned decision heads over the frozen audio models' embeddings.

The v3.0 anger / aggression decisions were hand-written threshold rules over
four numbers (arousal, dominance, valence, HuBERT P(anger)).  On long
recordings they fired 60-75 times per hour on calm and happy speech.  A head
is a small logistic regression over what the models already compute for every
5 s emotion window:

* ``w2v2``   — mean-pooled hidden state of the audeering wav2vec2 MSP-Dim model (1024)
* ``hubert`` — pooled HuBERT SUPERB-ER projector output (256)
* ``ast``    — AudioSet AST embedding averaged over the window (768)
* ``dims`` / ``cat`` / ``tags`` — the arousal/dominance/valence, HuBERT class
  probabilities and AudioSet group scores of the window

Heads are trained on the *dev* split only (``python -m evaluation.heads``) and
stored as small JSON files in ``heads/``.  When a head file is missing, a
required model is unavailable, or ``USE_LEARNED_HEADS=false``, the detectors
fall back to the v3.0 rules.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)

HEADS_DIR = Path(__file__).resolve().parent / 'heads'

DIM_NAMES = ('arousal', 'dominance', 'valence')
CAT_NAMES = ('ang', 'hap', 'neu', 'sad')
TAG_GROUPS = ('speech', 'speech_dominant', 'cry', 'scream', 'laughter', 'child_speech',
              'music', 'impact', 'silence')


# ---------------------------------------------------------------------------
# Window features (shared by training and production — keep them identical)
# ---------------------------------------------------------------------------
def _tag_ranges(tag_timeline, t0s: np.ndarray, window: float) -> List[Tuple[int, int]]:
    """Index range of tagger windows overlapping each [t0, t0 + window)."""
    starts = np.asarray(tag_timeline.starts)
    ends = starts + tag_timeline.window
    out = []
    for t0 in t0s:
        lo = int(np.searchsorted(ends, t0, side='right'))
        hi = int(np.searchsorted(starts, t0 + window, side='left'))
        out.append((lo, max(lo, hi)))
    return out


def window_features(emotion_timeline, tag_timeline, blocks: Sequence[str]) -> Optional[np.ndarray]:
    """[n_windows, d] feature matrix for the emotion windows, or None if a block is unavailable."""
    etl = emotion_timeline
    n = len(etl.starts)
    if n == 0:
        return None
    ranges = None
    if tag_timeline is not None and any(b in ('ast', 'tags') for b in blocks):
        ranges = _tag_ranges(tag_timeline, np.asarray(etl.starts), etl.window)
    parts: List[np.ndarray] = []
    for b in blocks:
        if b in ('w2v2', 'hubert'):
            e = etl.embeddings.get(b)
            if e is None or len(e) != n:
                return None
            parts.append(np.asarray(e, dtype=np.float32))
        elif b == 'ast':
            emb = getattr(tag_timeline, 'embeddings', None) if tag_timeline is not None else None
            if emb is None or len(emb) != len(tag_timeline.starts):
                return None
            rows = np.zeros((n, emb.shape[1]), dtype=np.float32)
            for i, (lo, hi) in enumerate(ranges):
                if hi > lo:
                    rows[i] = emb[lo:hi].astype(np.float32).mean(axis=0)
            parts.append(rows)
        elif b == 'dims':
            if not etl.has_dimensional:
                return None
            parts.append(np.stack([np.asarray(getattr(etl, k), dtype=np.float32) for k in DIM_NAMES], axis=1))
        elif b == 'cat':
            if not all(k in etl.categorical for k in CAT_NAMES):
                return None
            parts.append(np.stack([np.asarray(etl.categorical[k], dtype=np.float32) for k in CAT_NAMES], axis=1))
        elif b == 'tags':
            if tag_timeline is None:
                return None
            cols = np.zeros((n, len(TAG_GROUPS)), dtype=np.float32)
            for j, g in enumerate(TAG_GROUPS):
                s = tag_timeline.group_scores(g)
                for i, (lo, hi) in enumerate(ranges):
                    if hi > lo:
                        cols[i, j] = float(s[lo:hi].max())
            parts.append(cols)
        else:
            raise ValueError(f'unknown feature block {b!r}')
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Linear head
# ---------------------------------------------------------------------------
@dataclass
class LinearHead:
    name: str
    blocks: Tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    bias: float
    threshold: float                     # operating point chosen on the dev split
    meta: Dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = ((np.asarray(X, dtype=np.float32) - self.mean) / self.scale) @ self.coef + self.bias
        return (1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))).astype(np.float32)

    def to_json(self, path: Path) -> None:
        d = {'name': self.name, 'blocks': list(self.blocks), 'threshold': self.threshold,
             'bias': self.bias, 'meta': self.meta,
             'mean': [round(float(x), 6) for x in self.mean],
             'scale': [round(float(x), 6) for x in self.scale],
             'coef': [round(float(x), 6) for x in self.coef]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))

    @classmethod
    def from_json(cls, path: Path) -> 'LinearHead':
        d = json.loads(Path(path).read_text())
        return cls(name=d['name'], blocks=tuple(d['blocks']),
                   mean=np.asarray(d['mean'], dtype=np.float32),
                   scale=np.asarray(d['scale'], dtype=np.float32),
                   coef=np.asarray(d['coef'], dtype=np.float32),
                   bias=float(d['bias']), threshold=float(d['threshold']), meta=d.get('meta', {}))


_HEADS: Dict[str, Optional[LinearHead]] = {}


def get_head(name: str) -> Optional[LinearHead]:
    """The head stored in ``heads/<name>.json`` (cached), or None when absent / disabled."""
    if not config.advanced.use_learned_heads:
        return None
    if name not in _HEADS:
        path = Path(config.advanced.heads_dir or HEADS_DIR) / f'{name}.json'
        head = None
        if path.exists():
            try:
                head = LinearHead.from_json(path)
                logger.info('Learned head %r loaded (%s, threshold %.3f)', name, '+'.join(head.blocks),
                            head.threshold)
            except Exception as e:
                logger.warning('Learned head %s unreadable, using rules: %s', path, e)
        _HEADS[name] = head
    return _HEADS[name]


HEAD_NAMES = ('anger', 'scream')


def wants_embeddings() -> bool:
    """True when any installed head needs model embeddings (so timelines should keep them)."""
    for name in HEAD_NAMES:
        head = get_head(name)
        if head is not None and any(b in ('w2v2', 'hubert', 'ast') for b in head.blocks):
            return True
    return False


def threshold(name: str) -> float:
    """Operating point: the config override (``EMOTION_ANGER_HEAD_THRESHOLD`` /
    ``VIOLENCE_SCREAM_HEAD_THRESHOLD``) when set, else the dev-tuned value in the head file."""
    head = get_head(name)
    override = {'anger': config.advanced.anger_head_threshold,
                'scream': config.violence.scream_head_threshold}.get(name)
    if override is not None and not math.isnan(override):
        return float(override)
    return head.threshold if head is not None else float('nan')


def apply_heads(emotion_timeline, tag_timeline) -> Dict[str, np.ndarray]:
    """Score every installed head on the emotion windows; results land in ``emotion_timeline.heads``."""
    if emotion_timeline is None:
        return {}
    for name in HEAD_NAMES:
        if name in emotion_timeline.heads:
            continue
        head = get_head(name)
        if head is None:
            continue
        try:
            X = window_features(emotion_timeline, tag_timeline, head.blocks)
            if X is None:
                logger.info('Head %r skipped: a model it needs is unavailable; using rules', name)
                continue
            p = head.predict(X)
            if p.shape != (len(emotion_timeline.starts),) or not np.isfinite(p).all():
                raise ValueError(f'bad output shape {p.shape}')
        except Exception as e:      # a damaged / mismatched head file must never fail the analysis
            logger.warning('Head %r failed (%s); using the v3.0 rule', name, e)
            continue
        emotion_timeline.heads[name] = p
    return emotion_timeline.heads
