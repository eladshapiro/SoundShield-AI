"""Long-recording montage test: event-level precision / recall and false alarms per hour.

Builds synthetic "recordings" by concatenating real clips (with short silences)
so the production pipeline (timelines + detectors, exactly as ``main.py`` runs
them) can be scored on long audio, which the clip-level benchmark cannot do.

    .venv/bin/python -m evaluation.montage --minutes 20 --seed 0

Two montages are built from the *test* split only:
  * ``negative``  — calm speech, everyday sounds, laughter, happy/neutral speech
  * ``events``    — the same background with cries, screams and angry speech
                    inserted at known times
Detections are matched to inserted events by time overlap.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import datasets as D  # noqa: E402

logger = logging.getLogger('montage')
SR = 16000


def _load(path: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.5          # normalise level so the montage is not dominated by loud clips
    return y.astype(np.float32)


def build(minutes: float, seed: int, with_events: bool) -> Tuple[np.ndarray, List[Dict]]:
    rng = random.Random(seed)
    test = lambda clips: [c for c in clips if c.split == 'test']  # noqa: E731
    background = (test([c for c in D.index('esc50') if c.category != 'crying_baby'])
                  + test(D.index('librispeech'))
                  + test([c for c in D.index('ravdess') if c.category in ('neutral', 'calm', 'happy')])
                  + test([c for c in D.index('cremad') if c.category in ('neutral', 'happy')]))
    events = {
        'cry': test([c for c in D.index('esc50') if c.category == 'crying_baby']) + test(D.index('donateacry')),
        'scream': test([c for c in D.index('vivae') if c.category in ('anger', 'fear', 'pain')
                        and c.intensity in ('strong', 'peak')]),
        'anger': test([c for c in D.index('ravdess') if c.category == 'angry'])
                 + test([c for c in D.index('cremad') if c.category == 'anger']),
    }
    rng.shuffle(background)
    for v in events.values():
        rng.shuffle(v)
    target = int(minutes * 60 * SR)
    pieces: List[np.ndarray] = []
    labels: List[Dict] = []
    pos = 0
    bg_i = 0
    ev_iters = {k: iter(v) for k, v in events.items()}
    next_event_at = rng.uniform(20, 40) * SR if with_events else float('inf')
    while pos < target:
        if pos >= next_event_at:
            kind = rng.choice(list(events))
            clip = next(ev_iters[kind], None)
            if clip is None:
                continue
            y = _load(clip.path)
            labels.append({'kind': kind, 'start': pos / SR, 'end': (pos + len(y)) / SR,
                           'source': clip.group})
            pieces.append(y)
            pos += len(y)
            next_event_at = pos + rng.uniform(25, 60) * SR
        else:
            clip = background[bg_i % len(background)]
            bg_i += 1
            y = _load(clip.path)
            labels.append({'kind': 'background', 'start': pos / SR, 'end': (pos + len(y)) / SR,
                           'source': clip.group})
            pieces.append(y)
            pos += len(y)
        gap = np.zeros(int(rng.uniform(0.3, 1.5) * SR), dtype=np.float32)
        pieces.append(gap)
        pos += len(gap)
    return np.concatenate(pieces)[:target], labels


def match(detections: List[Tuple[float, float]], events: List[Dict], kind: str,
          tolerance: float = 2.0) -> Dict:
    evs = [e for e in events if e['kind'] == kind]
    background = [e for e in events if e['kind'] == 'background']
    other_events = [e for e in events if e['kind'] not in (kind, 'background')]
    hit = [False] * len(evs)
    tp = 0
    fp = 0
    false_by_source: Dict[str, int] = {}
    for ds, de in detections:
        ok = False
        for i, e in enumerate(evs):
            if ds < e['end'] + tolerance and de > e['start'] - tolerance:
                ok = True
                hit[i] = True
        if ok:
            tp += 1
        else:
            fp += 1
            src = 'unknown'
            for e in other_events + background:
                if ds < e['end'] and de > e['start']:
                    src = ('event:' + e['kind'] + '/' + e['source']) if e['kind'] != 'background' else e['source']
                    break
            false_by_source[src] = false_by_source.get(src, 0) + 1
    return {'events': len(evs), 'detected_events': int(sum(hit)),
            'recall': (sum(hit) / len(evs)) if evs else None,
            'detections': len(detections), 'true_detections': tp, 'false_detections': fp,
            'false_by_source': dict(sorted(false_by_source.items(), key=lambda x: -x[1]))}


def run_pipeline(path: str) -> Dict:
    from main import KindergartenRecordingAnalyzer
    analyzer = KindergartenRecordingAnalyzer(language='en', use_advanced=True)
    t0 = time.time()
    res = analyzer.analyze_audio_file(path)
    res['_seconds'] = time.time() - t0
    return res


def score(res: Dict, labels: List[Dict], minutes: float) -> Dict:
    hours = minutes / 60.0
    cries = [(c['start_time'], c['end_time']) for c in res['cry_segments']]
    viol = [(v['start_time'], v['end_time']) for v in res['violence_segments']]
    shout = [(v['start_time'], v['end_time']) for v in res['violence_segments'] if 'shouting' in v['violence_types']]
    aggr = [(v['start_time'], v['end_time']) for v in res['violence_segments'] if 'aggressive_tone' in v['violence_types']]
    emo = [(e['start_time'], e['end_time']) for e in res['concerning_emotions']]
    out = {
        'cry': match(cries, labels, 'cry'),
        'violence_any_vs_scream+anger': match(viol, [e for e in labels if e['kind'] in ('scream', 'anger')], 'scream'),
        'shouting_vs_scream': match(shout, labels, 'scream'),
        'aggressive_tone_vs_anger': match(aggr, labels, 'anger'),
        'concerning_emotion_vs_anger': match(emo, labels, 'anger'),
    }
    # any violence detection vs any scream/anger event
    both = [dict(e, kind='x') if e['kind'] in ('scream', 'anger') else e for e in labels]
    out['violence_any_vs_scream+anger'] = match(viol, both, 'x')
    for k, v in out.items():
        v['false_per_hour'] = v['false_detections'] / hours
    out['unanswered_cries'] = len(res['neglect_analysis'].get('unanswered_cries', []))
    out['cries_with_response'] = sum(1 for c in res['cry_with_responses'] if c.get('response_detected'))
    out['models_used'] = res.get('models_used')
    out['pipeline_seconds'] = res.get('_seconds')
    out['realtime_factor'] = (minutes * 60) / res['_seconds'] if res.get('_seconds') else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='montage')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    for noisy in ('httpx', 'httpcore', 'urllib3', 'filelock', 'huggingface_hub', 'transformers'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out_dir = ROOT / 'data' / 'montage'
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {'minutes': args.minutes, 'seed': args.seed}
    for name, with_events in (('negative', False), ('events', True)):
        audio, labels = build(args.minutes, args.seed, with_events)
        path = out_dir / f'{name}_{args.seed}.wav'
        sf.write(path, audio, SR)
        (out_dir / f'{name}_{args.seed}.json').write_text(json.dumps(labels, indent=1))
        logger.info('%s montage: %.1f min, %d events -> %s', name, len(audio) / SR / 60,
                    sum(1 for e in labels if e['kind'] != 'background'), path)
        res = run_pipeline(str(path))
        report[name] = score(res, labels, args.minutes)
        logger.info('%s: %s', name, json.dumps(report[name], indent=1))
    out = ROOT / 'evaluation' / 'results' / f'{args.tag}.json'
    out.write_text(json.dumps(report, indent=2))
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
