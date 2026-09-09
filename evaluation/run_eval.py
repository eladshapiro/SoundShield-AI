"""Run the real-audio evaluation suite.

Examples
--------
    .venv/bin/python -m evaluation.run_eval --tag baseline
    .venv/bin/python -m evaluation.run_eval --tag baseline --tasks cry,violence --backends heuristic
    .venv/bin/python -m evaluation.run_eval --tag smoke --limit 30
"""
from __future__ import annotations

import os
# One BLAS/OpenMP thread per worker process: the heuristic sweep is process-parallel.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import hashlib
import json
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import datasets as D  # noqa: E402
from evaluation.metrics import binary_metrics, group_rates, best_threshold  # noqa: E402

logger = logging.getLogger('evaluation')
RESULTS_DIR = ROOT / 'evaluation' / 'results'
CACHE_DIR = ROOT / 'data' / 'cache'

TASK_FIELDS = {          # task -> (pred field, score field) in runner output
    'cry': ('cry_pred', 'cry_score'),
    'violence': ('violence_pred', 'violence_score'),
    'anger': ('anger_pred', 'anger_score'),
    'speech': ('speech_pred', 'speech_score'),
}
# the tagger has no anger output; it is run on the anger clips only to provide the speech gate
TASK_FIELDS_BY_BACKEND = {('anger', 'tagger'): ('speech_pred', 'speech_score')}


def _config_hash() -> str:
    """Cache key for the heuristic sweep: only fields the heuristic detectors read."""
    from config import config
    sections = {}
    for k in ('audio', 'cry', 'violence', 'emotion', 'neglect'):
        d = asdict(getattr(config, k))
        # ML-only operating points (violence.aggressive_*) do not affect the heuristics
        sections[k] = {kk: vv for kk, vv in d.items() if not kk.startswith('aggressive_')}
    blob = json.dumps(sections, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


class JsonlCache:
    def __init__(self, name: str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.file = CACHE_DIR / f'{name}.jsonl'
        self.rows: Dict[str, Dict] = {}
        if self.file.exists():
            for line in self.file.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.rows[r['path']] = r
                except json.JSONDecodeError:
                    continue

    def put(self, row: Dict):
        self.rows[row['path']] = row
        with open(self.file, 'a') as fh:
            fh.write(json.dumps(row) + '\n')


def _heuristic_worker(path: str) -> Dict:
    import warnings
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    from evaluation.runners import run_heuristics
    try:
        return run_heuristics(path)
    except Exception as e:  # keep the sweep alive on odd files
        return {'path': path, 'error': f'{type(e).__name__}: {e}'}


def run_backend(backend: str, paths: List[str], workers: int, use_cache: bool) -> Dict[str, Dict]:
    """Return {path: runner_output} for every path, using/refreshing the cache."""
    if backend == 'heuristic':
        cache = JsonlCache(f'heuristic_{_config_hash()}')
    else:
        cache = JsonlCache(backend)
    todo = [p for p in paths if not (use_cache and p in cache.rows)]
    logger.info('%s: %d clips (%d cached, %d to run)', backend, len(paths), len(paths) - len(todo), len(todo))
    t0 = time.time()
    if todo:
        if backend == 'heuristic':
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as ex:
                futs = {ex.submit(_heuristic_worker, p): p for p in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    cache.put(f.result())
                    if i % 200 == 0:
                        logger.info('  heuristic %d/%d (%.0fs)', i, len(todo), time.time() - t0)
        elif backend == 'hubert':
            from evaluation.runners import HubertRunner
            runner = HubertRunner()
            for i, p in enumerate(todo, 1):
                cache.put(runner(p))
                if i % 200 == 0:
                    logger.info('  hubert %d/%d (%.0fs)', i, len(todo), time.time() - t0)
        elif backend.startswith('tagger'):
            from evaluation.runners import TaggerRunner
            runner = TaggerRunner()
            for i, p in enumerate(todo, 1):
                cache.put(runner(p))
                if i % 200 == 0:
                    logger.info('  tagger %d/%d (%.0fs)', i, len(todo), time.time() - t0)
        elif backend.startswith('emotion2'):
            from evaluation.runners import Emotion2Runner
            runner = Emotion2Runner()
            for i, p in enumerate(todo, 1):
                cache.put(runner(p))
                if i % 200 == 0:
                    logger.info('  emotion2 %d/%d (%.0fs)', i, len(todo), time.time() - t0)
        else:
            raise ValueError(backend)
    logger.info('%s done in %.0fs', backend, time.time() - t0)
    return {p: cache.rows[p] for p in paths if p in cache.rows}


def derive_prediction(task: str, backend: str, o: Dict) -> Tuple[int, float]:
    """(pred, score) for a cached runner row using the *current* config thresholds.

    Cached rows hold raw model outputs (group maxima, per-window emotion values),
    so re-deriving here keeps reports consistent with config.py without re-running
    the models.  The rules mirror the production detectors.
    """
    from config import config
    tcfg = config.tagger
    if backend == 'tagger':
        if task == 'cry':
            return int(o['cry_max'] >= tcfg.cry_threshold), float(o['cry_max'])
        if task == 'violence':
            return int(o['scream_max'] >= tcfg.scream_threshold), float(o['scream_max'])
        if task in ('speech', 'anger'):
            dominant = o['speech_max'] if o['speech_max'] > o['cry_max'] + tcfg.speech_over_cry_margin else 0.0
            return int(dominant >= tcfg.speech_threshold), float(dominant)
    if backend == 'emotion2':
        from emotion_models import anger_rule, aggressive_rule
        w = o.get('windows')
        if w and 'arousal' in w:
            anger = [anger_rule(p, a, d, v) for p, a, d, v in zip(w['p_ang'], w['arousal'], w['dominance'], w['valence'])]
            aggr = [aggressive_rule(a, d, v) for a, d, v in zip(w['arousal'], w['dominance'], w['valence'])]
        elif w:
            anger = [anger_rule(p, None, None, None) for p in w['p_ang']]
            aggr = []
        else:
            return int(o.get(TASK_FIELDS[task][0], 0)), float(o.get(TASK_FIELDS[task][1], 0.0))
        if task == 'anger':
            return int(any(f for f, _, _ in anger)), float(max((c for _, c, _ in anger), default=0.0))
        if task == 'violence':
            return int(any(f for f, _ in aggr)), float(max((c for _, c in aggr), default=0.0))
    pred_f, score_f = TASK_FIELDS_BY_BACKEND.get((task, backend), TASK_FIELDS[task])
    return int(o[pred_f]), float(o.get(score_f, o[pred_f]))


def fuse(task: str, tag_row: Dict, emo_row: Dict) -> Tuple[int, float]:
    """Production fusion: tagger events + speech-gated neural emotion (clip level)."""
    from config import config
    tcfg = config.tagger
    speech_ok = tag_row['speech_max'] >= tcfg.speech_threshold and \
        tag_row['speech_max'] > tag_row['cry_max'] + tcfg.speech_over_cry_margin
    if task == 'cry':
        return derive_prediction('cry', 'tagger', tag_row)
    if task == 'speech':
        return derive_prediction('speech', 'tagger', tag_row)
    if task == 'anger':
        p, c = derive_prediction('anger', 'emotion2', emo_row)
        return int(p and speech_ok), c if speech_ok else 0.0
    if task == 'violence':
        p_s, c_s = derive_prediction('violence', 'tagger', tag_row)
        p_a, c_a = derive_prediction('violence', 'emotion2', emo_row)
        p_a = int(p_a and speech_ok)
        return int(p_s or p_a), max(c_s, c_a if p_a else 0.0)
    raise ValueError(task)


def evaluate_task(task: str, backend: str, items: List[D.Labelled], outputs: Dict[str, Dict],
                  outputs2: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Metrics on all clips plus dev/test splits; threshold tuned on dev, applied to test."""
    rows = []
    errors = 0
    for it in items:
        o = outputs.get(it.clip.path)
        if not o or 'error' in o:
            errors += 1
            continue
        try:
            if backend == 'fusion':
                o2 = (outputs2 or {}).get(it.clip.path)
                if task in ('anger', 'violence') and (not o2 or 'error' in o2):
                    errors += 1
                    continue
                pred, score = fuse(task, o, o2 or {})
            else:
                pred, score = derive_prediction(task, backend, o)
        except KeyError:
            errors += 1
            continue
        rows.append((it.label, pred, score, it.clip.group, it.clip.split))
    y_true = [r[0] for r in rows]
    y_pred = [r[1] for r in rows]
    y_score = [r[2] for r in rows]
    groups = [r[3] for r in rows]
    m = binary_metrics(y_true, y_pred, y_score)
    m['errors'] = errors
    m['groups'] = group_rates(groups, y_true, y_pred)
    dev = [r for r in rows if r[4] == 'dev']
    test = [r for r in rows if r[4] == 'test']
    if dev and test:
        m['dev'] = binary_metrics([r[0] for r in dev], [r[1] for r in dev], [r[2] for r in dev])
        m['test'] = binary_metrics([r[0] for r in test], [r[1] for r in test], [r[2] for r in test])
        bt = best_threshold([r[0] for r in dev], [r[2] for r in dev])
        thr = bt['threshold']
        m['tuned_on_dev'] = {'threshold': thr, 'dev_f1': bt.get('f1', 0.0)}
        m['tuned_on_dev']['test'] = binary_metrics([r[0] for r in test],
                                                   [int(r[2] >= thr) for r in test],
                                                   [r[2] for r in test])
    m['best_threshold'] = best_threshold(y_true, y_score)
    return m


def evaluate_asr(items: List[D.Labelled], model_name: str, use_cache: bool, language: str = 'he') -> Dict[str, Any]:
    import jiwer
    from evaluation.runners import WhisperRunner, normalize_hebrew
    cache = JsonlCache(f'asr_{language}_' + model_name.replace('/', '__'))
    todo = [it.clip.path for it in items if not (use_cache and it.clip.path in cache.rows)]
    t0 = time.time()
    if todo:
        runner = WhisperRunner(model_name)
        for i, p in enumerate(todo, 1):
            cache.put(runner(p, language=language))
            if i % 50 == 0:
                logger.info('  asr[%s] %d/%d (%.0fs)', model_name, i, len(todo), time.time() - t0)
    refs, hyps = [], []
    for it in items:
        o = cache.rows.get(it.clip.path)
        if not o:
            continue
        refs.append(normalize_hebrew(it.clip.transcript))
        hyps.append(normalize_hebrew(o['hypothesis']))
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return {'model': model_name, 'n': len(refs), 'wer': wer, 'cer': cer,
            'seconds': time.time() - t0 if todo else None}


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Evaluation report — `{report['tag']}`", '',
             f"Generated {report['generated']} · config hash `{report['config_hash']}`", '']
    for task, per_backend in report['tasks'].items():
        if task in ('asr_he', 'asr_en'):
            title = 'Hebrew ASR (FLEURS he_il dev)' if task == 'asr_he' else 'English ASR (LibriSpeech dev-clean, 200 utt.)'
            lines += [f'## {title}', '', '| model | n | WER | CER |', '|---|---|---|---|']
            for r in per_backend:
                lines.append(f"| {r['model']} | {r['n']} | {r['wer']*100:.1f}% | {r['cer']*100:.1f}% |")
            lines.append('')
            continue
        lines += [f'## Task: {task}', '',
                  '| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | test F1 @ dev-tuned thr |',
                  '|---|---|---|---|---|---|---|---|---|']
        for backend, m in per_backend.items():
            auc = m.get('roc_auc')
            bt = m.get('tuned_on_dev', {}).get('test', {})
            bt = dict(bt, threshold=m.get('tuned_on_dev', {}).get('threshold', 0.0)) if bt else m.get('best_threshold', {})
            lines.append(f"| {backend} | {m['n_pos']} | {m['n_neg']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['f1']:.3f} | {m['fpr']:.3f} | {auc:.3f} | " if auc is not None else
                         f"| {backend} | {m['n_pos']} | {m['n_neg']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['f1']:.3f} | {m['fpr']:.3f} | n/a | ")
            lines[-1] += f"{bt.get('f1', 0):.3f} @ {bt.get('threshold', 0):.3f} |"
        lines.append('')
        for backend, m in per_backend.items():
            lines += [f'<details><summary>{task} / {backend}: per-group rates</summary>', '',
                      '| group | n | label | rate |', '|---|---|---|---|']
            for g, r in m['groups'].items():
                kind = 'recall' if r['label'] == 1 else 'FPR'
                lines.append(f"| {g} | {r['n']} | {r['label']} | {kind}={r['positive_rate']:.3f} |")
            lines += ['', '</details>', '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--tasks', default='cry,violence,anger,speech')
    ap.add_argument('--backends', default='heuristic',
                    help='comma list: heuristic, hubert, tagger, emotion2 (task-specific ones are skipped where n/a)')
    ap.add_argument('--asr-models', default='base,ivrit-ai/whisper-large-v3-turbo-ct2',
                    help='faster-whisper models to score on FLEURS he (only if asr_he in --tasks)')
    ap.add_argument('--asr-models-en', default='base,large-v3-turbo',
                    help='faster-whisper models to score on LibriSpeech dev-clean (only if asr_en in --tasks)')
    ap.add_argument('--neg-limit', type=int, default=None, help='cap negatives per task (stratified)')
    ap.add_argument('--limit', type=int, default=None, help='cap total clips per task (smoke test)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    for noisy in ('httpx', 'httpcore', 'urllib3', 'filelock', 'huggingface_hub'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    backends = [b.strip() for b in args.backends.split(',') if b.strip()]
    report: Dict[str, Any] = {'tag': args.tag, 'generated': time.strftime('%Y-%m-%d %H:%M'),
                              'config_hash': _config_hash(), 'tasks': {}}
    backend_applies = {
        'heuristic': {'cry', 'violence', 'anger', 'speech'},
        'hubert': {'anger'},
        'tagger': {'cry', 'violence', 'speech'},
        'emotion2': {'anger', 'violence'},
        'fusion': {'cry', 'violence', 'anger', 'speech'},     # production rules over tagger + emotion2
    }

    for task in tasks:
        items = D.build_task(task, neg_limit=args.neg_limit)
        if args.limit:
            pos = [i for i in items if i.label == 1][: args.limit // 2]
            neg = [i for i in items if i.label == 0][: args.limit - len(pos)]
            items = pos + neg
        if task in ('asr_he', 'asr_en'):
            lang = task.split('_')[1]
            models = args.asr_models if task == 'asr_he' else args.asr_models_en
            report['tasks'][task] = [evaluate_asr(items, m.strip(), not args.no_cache, language=lang)
                                     for m in models.split(',') if m.strip()]
            for r in report['tasks'][task]:
                logger.info('ASR %s: WER=%.3f CER=%.3f n=%d', r['model'], r['wer'], r['cer'], r['n'])
            continue
        paths = [i.clip.path for i in items]
        report['tasks'][task] = {}
        for backend in backends:
            if task not in backend_applies.get(backend, set()):
                continue
            if backend == 'fusion':
                outputs = run_backend('tagger', paths, args.workers, not args.no_cache)
                outputs2 = run_backend('emotion2', paths, args.workers, not args.no_cache) \
                    if task in ('anger', 'violence') else {}
                m = evaluate_task(task, backend, items, outputs, outputs2)
            else:
                outputs = run_backend(backend, paths, args.workers, not args.no_cache)
                m = evaluate_task(task, backend, items, outputs)
            report['tasks'][task][backend] = m
            logger.info('%s/%s: P=%.3f R=%.3f F1=%.3f FPR=%.3f AUC=%s (errors=%d)', task, backend,
                        m['precision'], m['recall'], m['f1'], m['fpr'],
                        f"{m['roc_auc']:.3f}" if 'roc_auc' in m else 'n/a', m['errors'])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'{args.tag}.json').write_text(json.dumps(report, indent=2))
    (RESULTS_DIR / f'{args.tag}.md').write_text(to_markdown(report))
    logger.info('wrote %s', RESULTS_DIR / f'{args.tag}.md')
    return report


if __name__ == '__main__':
    main()
