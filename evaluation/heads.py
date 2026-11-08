"""Train and evaluate the learned decision heads (``event_heads.py``) on real audio.

    .venv/bin/python -m evaluation.heads extract   # model embeddings for clips + montages (GPU, cached)
    .venv/bin/python -m evaluation.heads train     # cross-fitted on dev, writes heads/{anger,scream}.json
    .venv/bin/python -m evaluation.heads report    # test split + test montages -> evaluation/results/heads.md

Two heads:

* ``anger``  — angry adult speech (feeds ``concerning_emotions`` and the violence
  detector's ``aggressive_tone``)
* ``scream`` — distress screams: fear / pain / anger vocalisations, as opposed to
  joyful screams, kids at play, cries and everyday sounds (feeds ``shouting``)

Protocol (nothing from the test split is used for fitting or for choosing anything):

* The dev split is cut into two folds by speaker / source.  Each fold gets its own
  20-minute montages built only from that fold's clips (plain and "kindergarten"
  ones where half the background is kids playing), so a fold's predictions come
  from a head that never heard any of its clips (cross-fitting, no leakage through
  the montages).
* Feature blocks, regularisation and the operating point are chosen on those
  out-of-fold predictions; the shipped head is then refitted on all dev data.
* The test split (clip level) and the same test montages as ``evaluation.montage``
  (seeds 0 and 1, plus kindergarten versions) are only scored, against the v3.0
  rules on identical windows, through the production detector code.

Window labels.  Anger: RAVDESS ``angry`` / CREMA-D ``anger`` clips = 1, every other
emotion, read speech, cries, kids and everyday sounds = 0, VIVAE screams left out.
Scream: VIVAE anger / fear / pain at strong or peak intensity = 1 (lower intensities
left out), VIVAE achievement / pleasure / surprise and everything else = 0, strongly
shouted angry speech left out.  A montage window is positive when it overlaps an
inserted event of the head's kind by enough (anger 1 s, scream 0.5 s, or half the
clip), negative when it overlaps none, and ignored when it only grazes one or
overlaps the other head's events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import datasets as D  # noqa: E402
from evaluation import montage as M  # noqa: E402
from evaluation.metrics import binary_metrics  # noqa: E402

logger = logging.getLogger('heads')

CACHE = D.DATA_DIR / 'cache' / 'heads'
MONTAGE_DIR = D.DATA_DIR / 'montage'
RESULTS = ROOT / 'evaluation' / 'results'
ALL_BLOCKS = ('w2v2', 'hubert', 'ast', 'dims', 'cat', 'tags')
HEADS = ('anger', 'scream')
MINUTES = 20.0
DEV_EVENT_SEEDS = (100, 101, 102, 103)
DEV_NEG_SEEDS = (200,)
DEV_KG_SEEDS = (110, 111)
TEST_SEEDS = (0, 1)
KIDS_SHARE = 0.5          # "kindergarten" montages: half the background pieces are kids playing
MIN_OVERLAP = {'anger': 1.0, 'scream': 0.5}
OTHER = {'anger': 'scream', 'scream': 'anger'}


# ---------------------------------------------------------------------------
# Clips and folds
# ---------------------------------------------------------------------------
def source_key(c: D.Clip) -> str:
    """Speaker / recording source, so one person's clips never sit in two folds."""
    if c.dataset == 'esc50':
        return 'esc50:' + Path(c.path).stem.split('-')[1]          # freesound source id
    if c.dataset == 'donateacry':
        return 'donateacry:' + Path(c.path).stem
    return f'{c.dataset}:{c.speaker or Path(c.path).stem}'


def fold_of(c: D.Clip) -> int:
    return int(hashlib.md5(source_key(c).encode()).hexdigest(), 16) % 2


def clip_set() -> List[D.Clip]:
    """Every clip the cry/violence/anger/speech tasks use, all acted emotions, kids, Hebrew speech."""
    seen: Dict[str, D.Clip] = {}
    for task in ('anger', 'violence', 'speech', 'cry'):
        for it in D.build_task(task):
            seen[it.clip.path] = it.clip
    for name in ('ravdess', 'cremad', 'vivae', 'us8k_children'):
        for c in D.index(name):
            seen[c.path] = c
    for c in D.index('fleurs_he'):                    # test-only Hebrew negatives
        seen[c.path] = c
    return sorted(seen.values(), key=lambda c: c.path)


def clip_label(c: D.Clip, head: str = 'anger') -> Optional[int]:
    """Window label of a clip for ``head`` (None = left out of training and scoring)."""
    if head == 'anger':
        if c.dataset == 'vivae':
            return None
        return int((c.dataset == 'ravdess' and c.category == 'angry')
                   or (c.dataset == 'cremad' and c.category == 'anger'))
    if head == 'scream':
        if c.dataset == 'vivae':
            if c.category in ('anger', 'fear', 'pain'):
                return 1 if c.intensity in ('strong', 'peak') else None
            return 0
        if c.dataset == 'ravdess' and c.category == 'angry' and c.intensity == 'strong':
            return None
        return 0
    raise ValueError(head)


def _clip(m: Dict) -> D.Clip:
    return D.Clip(**{k: m[k] for k in D.Clip.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Models (production code paths)
# ---------------------------------------------------------------------------
class Models:
    def __init__(self):
        from audio_analyzer import AudioAnalyzer
        from audio_tagger import AudioTagger
        from emotion_models import EmotionAnalyzer
        self.audio = AudioAnalyzer()
        self.tagger = AudioTagger()
        self.emotion = EmotionAnalyzer()
        if not self.tagger.load():
            raise RuntimeError(f'tagger unavailable: {self.tagger.load_error}')
        if not (self.emotion.dim and self.emotion.dim.load() and self.emotion.cat and self.emotion.cat.load()):
            raise RuntimeError('emotion models unavailable')

    def timelines(self, path: str):
        audio, sr = self.audio.load_audio(path)       # same loading as main.py
        tl = self.tagger.tag(audio, sr, embeddings=True)
        et = self.emotion.timeline(audio, sr, embeddings=True)
        return tl, et


def _release_gpu() -> None:
    """Hand cached GPU memory back between recordings (other programs share this GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def gate_values(tl, et) -> np.ndarray:
    """Production speech gate per emotion window (speech_dominant max over the window)."""
    return np.array([tl.score('speech_dominant', s, min(s + et.window, et.duration)) for s in et.starts],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Timeline (de)serialisation for montages
# ---------------------------------------------------------------------------
def save_timelines(path: Path, tl, et) -> None:
    arrays = {'t_starts': tl.starts, 't_window': np.float64(tl.window), 't_probs': tl.probs.astype(np.float16),
              't_duration': np.float64(tl.duration), 't_emb': tl.embeddings,
              'e_starts': et.starts, 'e_window': np.float64(et.window), 'e_duration': np.float64(et.duration),
              'e_arousal': et.arousal, 'e_dominance': et.dominance, 'e_valence': et.valence}
    for k, v in et.categorical.items():
        arrays[f'e_cat_{k}'] = v
    for k, v in et.embeddings.items():
        arrays[f'e_emb_{k}'] = v.astype(np.float16)
    np.savez_compressed(path, labels=np.array(json.dumps(tl.labels)), **arrays)


def load_timelines(path: Path):
    from audio_tagger import TagTimeline
    from emotion_models import EmotionTimeline
    z = np.load(path, allow_pickle=False)
    tl = TagTimeline(starts=z['t_starts'], window=float(z['t_window']), probs=z['t_probs'].astype(np.float32),
                     labels=json.loads(str(z['labels'])), duration=float(z['t_duration']),
                     embeddings=z['t_emb'])
    cat = {k[6:]: z[k] for k in z.files if k.startswith('e_cat_')}
    emb = {k[6:]: z[k].astype(np.float32) for k in z.files if k.startswith('e_emb_')}
    et = EmotionTimeline(starts=z['e_starts'], window=float(z['e_window']), duration=float(z['e_duration']),
                         arousal=z['e_arousal'], dominance=z['e_dominance'], valence=z['e_valence'],
                         categorical=cat, embeddings=emb)
    return tl, et


def montage_specs() -> List[Dict]:
    """Dev montages per fold (training) and test montages (reporting); ``kg`` = with kids playing."""
    specs = []
    for fold in (0, 1):
        for seed in DEV_EVENT_SEEDS:
            specs.append({'name': f'dev{fold}_events_{seed}', 'split': 'dev', 'fold': fold, 'seed': seed,
                          'events': True, 'kids': 0.0})
        for seed in DEV_NEG_SEEDS:
            specs.append({'name': f'dev{fold}_negative_{seed}', 'split': 'dev', 'fold': fold, 'seed': seed,
                          'events': False, 'kids': 0.0})
        for seed in DEV_KG_SEEDS:
            specs.append({'name': f'dev{fold}_kg_events_{seed}', 'split': 'dev', 'fold': fold, 'seed': seed,
                          'events': True, 'kids': KIDS_SHARE})
    for seed in TEST_SEEDS:
        for kids in (0.0, KIDS_SHARE):
            for events in (False, True):
                kind = ('kg_' if kids else '') + ('events' if events else 'negative')
                specs.append({'name': f'test_{kind}_{seed}', 'split': 'test', 'fold': None, 'seed': seed,
                              'events': events, 'kids': kids})
    return specs


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def extract(args) -> None:
    import soundfile as sf
    from event_heads import window_features
    CACHE.mkdir(parents=True, exist_ok=True)
    MONTAGE_DIR.mkdir(parents=True, exist_ok=True)
    models = Models()

    # --- clips (incremental: only clips not in the cache yet) -------------------
    clip_npz = CACHE / 'clips.npz'
    X, gate, win_start, clip_idx, meta = [], [], [], [], []
    if clip_npz.exists() and not args.force:
        z = np.load(clip_npz)
        X, gate, win_start, clip_idx = [z['X']], [z['gate']], [z['win_start']], [z['clip_idx']]
        meta = json.loads((CACHE / 'clips.json').read_text())
    done = {m['path'] for m in meta}
    clips = [c for c in clip_set() if c.path not in done]
    if not clips:
        logger.info('clips cached: %s (%d clips)', clip_npz, len(meta))
    else:
        logger.info('extracting %d clips (%d already cached)', len(clips), len(meta))
        t0 = time.time()
        for i, c in enumerate(clips):
            try:
                tl, et = models.timelines(c.path)
                F = window_features(et, tl, ALL_BLOCKS)
            except Exception as e:                      # unreadable clip: skip, keep going
                logger.warning('skip %s: %s', c.path, e)
                continue
            X.append(F)
            gate.append(gate_values(tl, et))
            win_start.append(np.asarray(et.starts, dtype=np.float32))
            clip_idx.append(np.full(len(F), len(meta), dtype=np.int32))
            meta.append(dict(c.to_dict(), duration=float(et.duration), n_windows=int(len(F))))
            if (i + 1) % 500 == 0:
                logger.info('  %d/%d clips (%.0fs)', i + 1, len(clips), time.time() - t0)
        np.savez(clip_npz, X=np.concatenate(X), gate=np.concatenate(gate),
                 win_start=np.concatenate(win_start), clip_idx=np.concatenate(clip_idx))
        (CACHE / 'clips.json').write_text(json.dumps(meta))
        logger.info('clips done: %d windows in %.0fs', sum(len(x) for x in X), time.time() - t0)

    # --- montages ------------------------------------------------------------
    for spec in montage_specs():
        out = CACHE / f"{spec['name']}.npz"
        if out.exists() and not args.force:
            continue
        keep = None if spec['fold'] is None else (lambda c, f=spec['fold']: fold_of(c) == f)
        audio, labels = M.build(MINUTES, spec['seed'], spec['events'], split=spec['split'], keep=keep,
                                kids_share=spec['kids'])
        wav = MONTAGE_DIR / f"heads_{spec['name']}.wav"
        sf.write(wav, audio, M.SR)
        (CACHE / f"{spec['name']}.labels.json").write_text(json.dumps(labels))
        t0 = time.time()
        tl, et = models.timelines(str(wav))
        save_timelines(out, tl, et)
        _release_gpu()
        logger.info('montage %s: %d events, %d emotion windows (%.0fs)', spec['name'],
                    sum(1 for e in labels if e['kind'] != 'background'), len(et.starts), time.time() - t0)


# ---------------------------------------------------------------------------
# Window datasets
# ---------------------------------------------------------------------------
def montage_window_labels(starts: np.ndarray, window: float, labels: List[Dict], head: str = 'anger') -> np.ndarray:
    """1 = an event of the head's kind inside, 0 = none, -1 = ignore (grazes one / overlaps the other kind)."""
    y = np.zeros(len(starts), dtype=np.int8)
    mine = [e for e in labels if e['kind'] == head]
    other = [e for e in labels if e['kind'] == OTHER[head]]
    for i, s in enumerate(starts):
        e0, e1 = s, s + window
        best = 0.0
        for e in mine:
            ov = min(e1, e['end']) - max(e0, e['start'])
            if ov > 0:
                need = min(MIN_OVERLAP[head], 0.5 * (e['end'] - e['start']))
                best = max(best, 1.0 if ov >= need else 0.5)
        if best >= 1.0:
            y[i] = 1
        elif best > 0 or any(min(e1, e['end']) > max(e0, e['start']) for e in other):
            y[i] = -1
    return y


def load_clip_windows():
    z = np.load(CACHE / 'clips.npz')
    meta = json.loads((CACHE / 'clips.json').read_text())
    return z['X'], z['gate'], z['win_start'], z['clip_idx'], meta


def block_slices() -> Dict[str, slice]:
    dims = {'w2v2': 1024, 'hubert': 256, 'ast': 768, 'dims': 3, 'cat': 4, 'tags': 9}
    out, i = {}, 0
    for b in ALL_BLOCKS:
        out[b] = slice(i, i + dims[b])
        i += dims[b]
    return out


def select(X: np.ndarray, blocks: Sequence[str]) -> np.ndarray:
    sl = block_slices()
    return np.concatenate([X[:, sl[b]] for b in blocks], axis=1)


def montage_windows(name: str, head: str = 'anger'):
    from event_heads import window_features
    tl, et = load_timelines(CACHE / f'{name}.npz')
    labels = json.loads((CACHE / f'{name}.labels.json').read_text())
    X = window_features(et, tl, ALL_BLOCKS)
    y = montage_window_labels(np.asarray(et.starts), et.window, labels, head)
    return X, y, gate_values(tl, et), tl, et, labels


def dev_training_data(head: str = 'anger'):
    """(X, y, fold, gate, source) over dev clip windows + dev montage windows (ignored windows dropped)."""
    X, gate, _, clip_idx, meta = load_clip_windows()
    lab = np.array([clip_label(_clip(m), head) for m in meta], dtype=object)
    folds = np.array([fold_of(_clip(m)) for m in meta])
    split = np.array([m['split'] for m in meta])
    keep = np.array([split[ci] == 'dev' and lab[ci] is not None for ci in clip_idx])
    Xs = [X[keep]]
    ys = [np.array([lab[ci] for ci in clip_idx[keep]], dtype=np.int8)]
    fs = [folds[clip_idx[keep]]]
    gs = [gate[keep]]
    src = ['clip'] * int(keep.sum())
    for spec in montage_specs():
        if spec['split'] != 'dev':
            continue
        Xm, ym, gm, *_ = montage_windows(spec['name'], head)
        k = ym >= 0
        Xs.append(Xm[k]); ys.append(ym[k]); fs.append(np.full(int(k.sum()), spec['fold'])); gs.append(gm[k])
        src += ['montage'] * int(k.sum())
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs), np.concatenate(gs), np.array(src)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-6] = 1.0
    clf = LogisticRegression(C=C, class_weight='balanced', max_iter=3000)
    clf.fit((X - mean) / scale, y)
    return mean.astype(np.float32), scale.astype(np.float32), clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def predict(params, X: np.ndarray) -> np.ndarray:
    mean, scale, coef, bias = params
    z = ((X - mean) / scale) @ coef + bias
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def oof_predictions(X, y, folds, C):
    p = np.zeros(len(y))
    for f in (0, 1):
        tr, te = folds != f, folds == f
        p[te] = predict(fit(X[tr], y[tr], C), X[te])
    return p


# ---------------------------------------------------------------------------
# Production replay on cached timelines
# ---------------------------------------------------------------------------
@contextmanager
def heads_state(et, heads: Optional[Dict[str, np.ndarray]], thresholds: Optional[Dict[str, float]] = None):
    """Run detector code with given head outputs (``None`` = the v3.0 rules, all heads off)."""
    from config import config
    adv, vio = config.advanced, config.violence
    saved = (adv.use_learned_heads, adv.anger_head_threshold, vio.scream_head_threshold)
    et.heads = {}
    try:
        if heads is None:
            adv.use_learned_heads = False
        else:
            et.heads.update({k: np.asarray(v, dtype=np.float32) for k, v in heads.items()})
            thresholds = thresholds or {}
            if 'anger' in thresholds:
                adv.anger_head_threshold = float(thresholds['anger'])
            if 'scream' in thresholds:
                vio.scream_head_threshold = float(thresholds['scream'])
        yield
    finally:
        adv.use_learned_heads, adv.anger_head_threshold, vio.scream_head_threshold = saved
        et.heads = {}


_ANALYZER = None
_VIOLENCE = None


def replay_concerning(tl, et, heads, thresholds=None) -> List[Tuple[float, float]]:
    """Production ``concerning_emotions_from_timeline`` (anger)."""
    global _ANALYZER
    from advanced_analyzer import AdvancedAnalyzer
    if _ANALYZER is None:
        _ANALYZER = AdvancedAnalyzer(use_whisper=False, use_transformer_emotion=False)
    with heads_state(et, heads, thresholds):
        res = _ANALYZER.concerning_emotions_from_timeline(et, tl)
    return [(r['start_time'], r['end_time']) for r in res]


def replay_violence(audio, sr, tl, et, heads, thresholds=None) -> List[Dict]:
    """Production ``ViolenceDetector`` ML path (shouting / aggressive tone / impacts)."""
    global _VIOLENCE
    from violence_detector import ViolenceDetector
    if _VIOLENCE is None:
        _VIOLENCE = ViolenceDetector()
    with heads_state(et, heads, thresholds):
        return _VIOLENCE._detect_violence_segments_ml(audio, sr, tl, et)


def montage_audio(name: str):
    from audio_analyzer import AudioAnalyzer
    return AudioAnalyzer().load_audio(str(MONTAGE_DIR / f'heads_{name}.wav'))


def montage_score(dets, labels, kind: str, minutes=MINUTES) -> Dict:
    m = M.match(dets, labels, kind)
    m['false_per_hour'] = m['false_detections'] / (minutes / 60.0)
    return m


def shouting(segs: List[Dict]) -> List[Tuple[float, float]]:
    return [(s['start_time'], s['end_time']) for s in segs if 'shouting' in s['violence_types']]


def dev_montage_curve(head: str, blocks, C, thresholds) -> List[Dict]:
    """Cross-fitted montage FA/hour + event recall per threshold (fold f scored by a head fitted on the other fold)."""
    X, y, folds, gate, _ = dev_training_data(head)
    Xb = select(X, blocks)
    params = {f: fit(Xb[folds != f], y[folds != f], C) for f in (0, 1)}
    per_thr = {t: {'fp': 0, 'events': 0, 'hit': 0, 'minutes': 0.0} for t in thresholds}
    for spec in montage_specs():
        if spec['split'] != 'dev':
            continue
        Xm, ym, gm, tl, et, labels = montage_windows(spec['name'], head)
        p = predict(params[spec['fold']], select(Xm, blocks))
        audio = sr = None
        if head == 'scream':
            audio, sr = montage_audio(spec['name'])
        for t in thresholds:
            if head == 'anger':
                dets = replay_concerning(tl, et, {'anger': p}, {'anger': t})
            else:
                dets = shouting(replay_violence(audio, sr, tl, et, {'scream': p}, {'scream': t}))
            s = montage_score(dets, labels, head)
            per_thr[t]['fp'] += s['false_detections']
            per_thr[t]['events'] += s['events']
            per_thr[t]['hit'] += s['detected_events']
            per_thr[t]['minutes'] += MINUTES
    out = []
    for t in thresholds:
        r = per_thr[t]
        out.append({'threshold': t, 'false_per_hour': r['fp'] / (r['minutes'] / 60),
                    'recall': r['hit'] / max(1, r['events']), 'events': r['events']})
    return out


BLOCK_SETS = {
    'anger': [
        ('w2v2', 'hubert', 'ast', 'dims', 'cat', 'tags'),
        ('w2v2', 'ast', 'dims', 'tags'),
        ('w2v2', 'hubert', 'dims', 'cat'),
        ('w2v2', 'dims'),
        ('hubert', 'cat'),
        ('ast', 'tags'),
        ('dims', 'cat', 'tags'),
    ],
    'scream': [
        ('w2v2', 'hubert', 'ast', 'dims', 'cat', 'tags'),
        ('w2v2', 'ast', 'dims', 'tags'),
        ('ast', 'tags'),
        ('w2v2', 'dims', 'tags'),
        ('ast', 'dims', 'tags'),
        ('dims', 'cat', 'tags'),
    ],
}
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)


def train_head(head: str, target_recall: float, out: Optional[Path]) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from event_heads import HEADS_DIR, LinearHead
    from config import config
    X, y, folds, gate, src = dev_training_data(head)
    logger.info('[%s] dev windows: %d (%d positive), clip %d / montage %d', head, len(y), int(y.sum()),
                int((src == 'clip').sum()), int((src == 'montage').sum()))
    # anger is only ever scored on speech-gated windows; screams on every window
    scored = gate >= config.tagger.speech_threshold if head == 'anger' else np.ones(len(y), dtype=bool)
    results = []
    for blocks in BLOCK_SETS[head]:
        Xb = select(X, blocks)
        for C in C_GRID:
            p = oof_predictions(Xb, y, folds, C)
            ap = average_precision_score(y[scored], p[scored])
            auc = roc_auc_score(y[scored], p[scored])
            results.append({'blocks': blocks, 'C': C, 'ap': ap, 'auc': auc})
            logger.info('  [%s] %-40s C=%-6g  OOF AP=%.3f AUC=%.3f', head, '+'.join(blocks), C, ap, auc)
    best = max(results, key=lambda r: r['ap'])
    logger.info('[%s] best: %s', head, best)
    thresholds = [round(float(t), 3) for t in np.arange(0.30, 0.995, 0.025)]
    curve = dev_montage_curve(head, best['blocks'], best['C'], thresholds)
    for r in curve:
        logger.info('  [%s] thr=%.3f  FA/h=%5.1f  recall=%.2f (%d events)', head, r['threshold'],
                    r['false_per_hour'], r['recall'], r['events'])
    # operating point: lowest false-alarm rate that keeps >= target recall on the dev montages
    ok = [r for r in curve if r['recall'] >= target_recall]
    op = min(ok, key=lambda r: (r['false_per_hour'], -r['recall'])) if ok else max(curve, key=lambda r: r['recall'])
    logger.info('[%s] operating point: %s', head, op)
    Xb = select(X, best['blocks'])
    mean, scale, coef, bias = fit(Xb, y, best['C'])
    models = {'w2v2': config.advanced.dim_emotion_model, 'hubert': config.advanced.hubert_model,
              'ast': config.tagger.model}
    model = LinearHead(name=head, blocks=tuple(best['blocks']), mean=mean, scale=scale, coef=coef, bias=bias,
                       threshold=float(op['threshold']),
                       meta={'trained': time.strftime('%Y-%m-%d'), 'C': best['C'],
                             'dev_windows': int(len(y)), 'dev_positive': int(y.sum()),
                             'oof_ap': round(best['ap'], 4), 'oof_auc': round(best['auc'], 4),
                             'dev_montage_operating_point': op, 'target_recall': target_recall,
                             'models': {b: models[b] for b in ('w2v2', 'hubert', 'ast')},
                             'protocol': 'evaluation/heads.py (dev split only; 2 speaker folds, cross-fitted)'})
    path = Path(out or HEADS_DIR / f'{head}.json')
    model.to_json(path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'heads_train_{head}.json').write_text(json.dumps(
        {'selection': [dict(r, blocks='+'.join(r['blocks'])) for r in results],
         'best': dict(best, blocks='+'.join(best['blocks'])),
         'dev_montage_curve': curve, 'operating_point': op}, indent=1))
    logger.info('[%s] wrote %s', head, path)


def train(args) -> None:
    for head in args.heads:
        target = args.target_recall if args.target_recall is not None else TARGET_RECALL[head]
        train_head(head, target, Path(args.out) if args.out else None)


TARGET_RECALL = {'anger': 0.80, 'scream': 0.80}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _clip_tables(name: str, meta, clip_idx, gated, win_rule, win_rule_score, win_head, win_head_score,
                 task_label) -> Dict:
    """Clip-level metrics (test split) for one head vs the v3.0 rule, plus per-group rates."""
    n_clips = len(meta)

    def clip_level(win_flag, win_score):
        flag = np.zeros(n_clips, dtype=bool)
        score = np.zeros(n_clips)
        np.logical_or.at(flag, clip_idx, win_flag & gated)
        np.maximum.at(score, clip_idx, np.where(gated, win_score, 0.0))
        return flag, score

    rule_flag, rule_score = clip_level(win_rule, win_rule_score)
    head_flag, head_score = clip_level(win_head, win_head_score)
    idx = [i for i, m in enumerate(meta) if m['split'] == 'test' and task_label(_clip(m)) is not None]
    y = [task_label(_clip(meta[i])) for i in idx]
    out: Dict = {'metrics': {}, 'groups': {}}
    for dec, flag, score in (('v3.0 rules', rule_flag, rule_score), ('learned head', head_flag, head_score)):
        out['metrics'][dec] = binary_metrics(y, [int(flag[i]) for i in idx], [float(score[i]) for i in idx])
    for i, lab in zip(idx, y):
        g = out['groups'].setdefault(meta[i]['group'], {'n': 0, 'label': lab, 'v3.0 rules': 0, 'learned head': 0})
        g['n'] += 1
        g['v3.0 rules'] += int(rule_flag[i])
        g['learned head'] += int(head_flag[i])
    return out


def report(args) -> None:
    from config import config
    from emotion_models import anger_rule
    from event_heads import HEADS_DIR, LinearHead
    heads = {h: LinearHead.from_json(Path(HEADS_DIR / f'{h}.json')) for h in HEADS
             if (HEADS_DIR / f'{h}.json').exists()}
    sl = block_slices()
    X, gate, _, clip_idx, meta = load_clip_windows()
    out: Dict = {'heads': {h: {'blocks': list(m.blocks), 'threshold': m.threshold, 'meta': m.meta}
                           for h, m in heads.items()}, 'clips': {}, 'montages': {}}

    # --- clip level -------------------------------------------------------------
    dims, cat, tags = X[:, sl['dims']], X[:, sl['cat']], X[:, sl['tags']]
    if 'anger' in heads:
        h = heads['anger']
        p = h.predict(select(X, h.blocks))
        rr = [anger_rule(float(c[0]), float(d[0]), float(d[1]), float(d[2])) for d, c in zip(dims, cat)]
        rule_flag, rule_score = np.array([r[0] for r in rr]), np.array([r[1] for r in rr])
        gated = gate >= config.tagger.speech_threshold
        task = {it.clip.path: it.label for it in D.build_task('anger')}     # v3.0 report's anger task
        out['clips']['anger_v30_task'] = _clip_tables(
            'anger', meta, clip_idx, gated, rule_flag, rule_score, p >= h.threshold, p,
            lambda c: task.get(c.path))
        out['clips']['anger'] = _clip_tables(
            'anger', meta, clip_idx, gated, rule_flag, rule_score, p >= h.threshold, p,
            lambda c: clip_label(c, 'anger'))
    if 'scream' in heads:
        from event_heads import TAG_GROUPS
        h = heads['scream']
        p = h.predict(select(X, h.blocks))
        scream_tag = tags[:, TAG_GROUPS.index('scream')]
        out['clips']['scream'] = _clip_tables(
            'scream', meta, clip_idx, np.ones(len(p), dtype=bool),
            scream_tag >= config.tagger.scream_threshold, scream_tag, p >= h.threshold, p,
            lambda c: clip_label(c, 'scream'))

    # --- montages: full production detectors, heads on vs off ---------------------
    for spec in montage_specs():
        if spec['split'] != 'test':
            continue
        name = spec['name']
        _, _, _, tl, et, labels = montage_windows(name)
        audio, sr = montage_audio(name)
        hp = {}
        for hname, model in heads.items():
            from event_heads import window_features
            Xm = window_features(et, tl, model.blocks)
            hp[hname] = model.predict(Xm)
        thr = {hname: model.threshold for hname, model in heads.items()}
        row = {}
        for dec, hd in (('v3.0 rules', None), ('learned heads', hp)):
            emo = replay_concerning(tl, et, hd, thr)
            viol = replay_violence(audio, sr, tl, et, hd, thr)
            both = [dict(e, kind='x') if e['kind'] in ('scream', 'anger') else e for e in labels]
            row[dec] = {
                'anger (concerning emotion)': montage_score(emo, labels, 'anger'),
                'shouting vs screams': montage_score(shouting(viol), labels, 'scream'),
                'aggressive tone vs anger': montage_score(
                    [(s['start_time'], s['end_time']) for s in viol if 'aggressive_tone' in s['violence_types']],
                    labels, 'anger'),
                'any violence vs screams+anger': montage_score(
                    [(s['start_time'], s['end_time']) for s in viol], both, 'x'),
            }
        out['montages'][name] = row
        logger.info('montage %s scored', name)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'heads.json').write_text(json.dumps(out, indent=1, default=float))
    md = to_markdown(out)
    (RESULTS / 'heads.md').write_text(md)
    print(md)


def _rate(d: Dict, key: str) -> str:
    return f"{d[key] / d['n']:.3f}" if d['n'] else '-'


def to_markdown(r: Dict) -> str:
    L = ['# Learned decision heads — held-out results', '',
         'Heads are fitted and their thresholds chosen on the **dev** split only (two speaker folds, '
         'cross-fitted, own montages per fold; `python -m evaluation.heads train`). Everything below is the '
         '**test split** and the test montages, scored through the production detector code.', '']
    for h, d in r['heads'].items():
        m = d['meta']
        L.append(f"* `heads/{h}.json`: blocks `{'+'.join(d['blocks'])}`, threshold {d['threshold']:.3f} "
                 f"(dev montages: {m['dev_montage_operating_point']['false_per_hour']:.1f} false/h at recall "
                 f"{m['dev_montage_operating_point']['recall']:.2f}), {m['dev_windows']} dev windows, "
                 f"OOF AP {m['oof_ap']:.3f}")
    titles = {'anger_v30_task': ('Angry speech, the v3.0 anger task (test clips, speech-gated, any window)',
                                 'Same clips as `final.md`: RAVDESS / CREMA-D angry vs neutral, calm, happy, and '
                                 'LibriSpeech. v3.0 rule: `anger_rule`'),
              'anger': ('Angry speech, every test clip (speech-gated, any window)',
                        'All other emotions, cries, kids at play, everyday sounds and Hebrew speech as '
                        'negatives too. v3.0 rule: `anger_rule`'),
              'scream': ('Distress screams, test clips (any window)',
                         'v3.0 rule: AudioSet scream group >= `TAGGER_SCREAM_THRESHOLD`')}
    for h, t in r['clips'].items():
        L += ['', f'## {titles[h][0]}', '', titles[h][1], '',
              '| decision | pos | neg | precision | recall | F1 | FPR | ROC-AUC |', '|---|---|---|---|---|---|---|---|']
        for dec, m in t['metrics'].items():
            L.append(f"| {dec} | {m['n_pos']} | {m['n_neg']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1']:.3f} | {m['fpr']:.3f} | {m.get('roc_auc', float('nan')):.3f} |")
        L += ['', '| group | n | rate | v3.0 rules | learned head |', '|---|---|---|---|---|']
        for g, d in sorted(t['groups'].items()):
            L.append(f"| {g} | {d['n']} | {'recall' if d['label'] == 1 else 'FPR'} | {_rate(d, 'v3.0 rules')} | "
                     f"{_rate(d, 'learned head')} |")
    L += ['', '## 20-minute test montages (production detectors, heads off vs on)', '',
          '`kg` = kindergarten montage: half the background pieces are UrbanSound8K kids at play.', '',
          '| montage | detector | v3.0 rules: found / events | v3.0 false/h | heads: found / events | heads false/h |',
          '|---|---|---|---|---|---|']
    totals: Dict[str, Dict[str, List[float]]] = {}
    for name, row in r['montages'].items():
        for det in row['v3.0 rules']:
            a, b = row['v3.0 rules'][det], row['learned heads'][det]
            fa = f"{a['detected_events']} / {a['events']}" if a['events'] else '-'
            fb = f"{b['detected_events']} / {b['events']}" if b['events'] else '-'
            L.append(f"| {name} | {det} | {fa} | {a['false_per_hour']:.1f} | {fb} | {b['false_per_hour']:.1f} |")
            t = totals.setdefault(det, {'ev': [0, 0], 'hit': [0, 0], 'fp': [0, 0], 'n': [0, 0]})
            for k, mm in enumerate((a, b)):
                t['ev'][k] += mm['events']
                t['hit'][k] += mm['detected_events']
                t['fp'][k] += mm['false_detections']
                t['n'][k] += 1
    L += ['', '**All test montages together** (8 × 20 min):', '',
          '| detector | v3.0 recall | v3.0 false/h | heads recall | heads false/h |', '|---|---|---|---|---|']
    for det, t in totals.items():
        hours = [n * MINUTES / 60 for n in t['n']]
        rec = [f"{t['hit'][k] / t['ev'][k]:.2f} ({t['hit'][k]}/{t['ev'][k]})" if t['ev'][k] else '-' for k in (0, 1)]
        L.append(f"| {det} | {rec[0]} | {t['fp'][0] / hours[0]:.1f} | {rec[1]} | {t['fp'][1] / hours[1]:.1f} |")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['extract', 'train', 'report', 'all'])
    ap.add_argument('--heads', default=','.join(HEADS), help='comma list of heads to train')
    ap.add_argument('--force', action='store_true', help='recompute cached embeddings')
    ap.add_argument('--target-recall', type=float, default=None,
                    help='dev-montage event recall the operating point must keep (default 0.80)')
    ap.add_argument('--out', default=None, help='head file to write (single head only)')
    args = ap.parse_args(argv)
    args.heads = [h.strip() for h in args.heads.split(',') if h.strip()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    for noisy in ('httpx', 'httpcore', 'urllib3', 'filelock', 'huggingface_hub', 'transformers',
                  'advanced_analyzer', 'violence_detector', 'event_heads'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.stage in ('extract', 'all'):
        extract(args)
    if args.stage in ('train', 'all'):
        train(args)
    if args.stage in ('report', 'all'):
        report(args)


if __name__ == '__main__':
    main()
