"""Threshold / fusion-rule sweep over cached runner outputs.

Rules are tuned on the ``dev`` split and reported on ``test`` so the chosen
operating points are not overfitted to the clips they are reported on.

    .venv/bin/python -m evaluation.fusion
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import datasets as D            # noqa: E402
from evaluation.metrics import binary_metrics    # noqa: E402
from evaluation.run_eval import JsonlCache       # noqa: E402


def load(backend: str) -> Dict[str, Dict]:
    return JsonlCache(backend).rows


def rows_for(task: str, caches: List[Dict[str, Dict]]):
    out = []
    for it in D.build_task(task):
        feats = {}
        ok = True
        for c in caches:
            r = c.get(it.clip.path)
            if not r or 'error' in r:
                ok = False
                break
            feats.update(r)
        if ok:
            out.append((it, feats))
    return out


def evaluate(rows, rule: Callable[[Dict], bool], split: str) -> Dict:
    sel = [(it, f) for it, f in rows if it.clip.split == split]
    y = [it.label for it, _ in sel]
    p = [int(rule(f)) for _, f in sel]
    return binary_metrics(y, p)


def sweep(rows, make_rule: Callable[..., Callable[[Dict], bool]], grid: Dict[str, List[float]],
          min_precision: float = 0.0) -> Tuple[Dict, Dict, Dict]:
    """Grid search on dev, report the winner on test."""
    import itertools
    keys = list(grid)
    best = None
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        m = evaluate(rows, make_rule(**params), 'dev')
        if m['precision'] < min_precision:
            continue
        if best is None or m['f1'] > best[1]['f1']:
            best = (params, m)
    params, dev_m = best
    test_m = evaluate(rows, make_rule(**params), 'test')
    return params, dev_m, test_m


def fmt(m: Dict) -> str:
    return f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.3f} (n+={m['n_pos']}, n-={m['n_neg']})"


def group_report(rows, rule, split='test') -> str:
    lines = []
    by = {}
    for it, f in rows:
        if it.clip.split != split:
            continue
        by.setdefault(it.clip.group, []).append((it.label, int(rule(f))))
    for g, lst in sorted(by.items()):
        y = np.array([a for a, _ in lst]); p = np.array([b for _, b in lst])
        kind = 'recall' if y.mean() > 0.5 else 'FPR'
        lines.append(f"      {g:28s} n={len(lst):4d} {kind}={p.mean():.3f}")
    return '\n'.join(lines)


def main():
    tagger, emo, heur = load('tagger'), load('emotion2'), None
    report = {}

    # ---------------- cry: tagger cry score ----------------
    rows = rows_for('cry', [tagger])
    grid = {'thr': [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]}
    rule = lambda thr: (lambda f: f['cry_max'] >= thr)
    params, dev_m, test_m = sweep(rows, rule, grid, min_precision=0.9)
    print(f"CRY   tagger cry_max>=thr        -> {params}  dev {fmt(dev_m)}\n{'':36s}test {fmt(test_m)}")
    print(group_report(rows, rule(**params)))
    print("      threshold curve (test):")
    for thr in grid['thr']:
        m = evaluate(rows, rule(thr), 'test')
        hn = [f for it, f in rows if it.clip.split == 'test' and it.clip.group == 'esc50/human_nonspeech' and it.label == 0]
        fpr_hn = np.mean([f['cry_max'] >= thr for f in hn]) if hn else 0
        print(f"        thr={thr:<5} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.4f} FPR(laugh/cough/..)={fpr_hn:.3f}")
    rule_i = lambda thr: (lambda f: f['infant_cry_max'] >= thr)
    pi, di, ti = sweep(rows, rule_i, grid, min_precision=0.9)
    print(f"CRY   tagger infant_cry_max>=thr -> {pi}  dev {fmt(di)}\n{'':36s}test {fmt(ti)}")
    report['cry'] = {'params': params, 'dev': dev_m, 'test': test_m, 'infant_only': (pi, di, ti)}

    # ---------------- speech / adult response ----------------
    rows = rows_for('speech', [tagger])
    grid = {'thr': [0.3, 0.4, 0.5, 0.6, 0.7]}
    rule = lambda thr: (lambda f: f['speech_max'] >= thr)
    params, dev_m, test_m = sweep(rows, rule, grid)
    print(f"\nSPEECH tagger speech_max>=thr     -> {params}  dev {fmt(dev_m)}\n{'':36s}test {fmt(test_m)}")
    rule2 = lambda thr, margin: (lambda f: f['speech_max'] >= thr and f['speech_max'] > f['cry_max'] + margin)
    params2, dev2, test2 = sweep(rows, rule2, {'thr': [0.3, 0.4, 0.5, 0.6, 0.7], 'margin': [0.0, 0.1, 0.2]})
    print(f"SPEECH speech>=thr & speech>cry+m -> {params2}  dev {fmt(dev2)}\n{'':36s}test {fmt(test2)}")
    print(group_report(rows, rule2(**params2)))
    report['speech'] = {'params': params2, 'dev': dev2, 'test': test2}

    # ---------------- anger: hubert / dims / fusion ----------------
    rows = rows_for('anger', [tagger, emo])
    r_h = lambda thr: (lambda f: f.get('p_ang_max', 0) >= thr)
    p1, d1, t1 = sweep(rows, r_h, {'thr': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    print(f"\nANGER hubert p_ang>=thr           -> {p1}  dev {fmt(d1)}\n{'':36s}test {fmt(t1)}")
    r_d = lambda a, d: (lambda f: f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d)
    p2, d2, t2 = sweep(rows, r_d, {'a': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], 'd': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]})
    print(f"ANGER arousal>=a & dominance>=d   -> {p2}  dev {fmt(d2)}\n{'':36s}test {fmt(t2)}")
    r_f = lambda thr, a, d: (lambda f: f.get('p_ang_max', 0) >= thr or
                             (f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d))
    p3, d3, t3 = sweep(rows, r_f, {'thr': [0.4, 0.5, 0.6, 0.7, 0.8, 1.1], 'a': [0.6, 0.65, 0.7, 0.75, 0.8], 'd': [0.55, 0.6, 0.65, 0.7, 0.75]})
    print(f"ANGER hubert OR dims              -> {p3}  dev {fmt(d3)}\n{'':36s}test {fmt(t3)}")
    # valence: anger is low-valence, excited/happy speech is high-valence
    r_v = lambda a, d, v: (lambda f: f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d
                           and f.get('valence_mean', 1) <= v)
    p4, d4, t4 = sweep(rows, r_v, {'a': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 'd': [0.6, 0.65, 0.7, 0.75, 0.8],
                                   'v': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0]})
    print(f"ANGER dims & valence<=v           -> {p4}  dev {fmt(d4)}\n{'':36s}test {fmt(t4)}")
    print(group_report(rows, r_v(**p4)))
    # dims AND (hubert anger OR low valence): two independent votes for anger
    r_w = lambda a, d, v, thr: (lambda f: f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d
                                and (f.get('valence_mean', 1) <= v or f.get('p_ang_max', 0) >= thr))
    p5, d5, t5 = sweep(rows, r_w, {'a': [0.5, 0.55, 0.6, 0.65, 0.7], 'd': [0.6, 0.65, 0.7, 0.75, 0.8],
                                   'v': [0.25, 0.3, 0.35, 0.4], 'thr': [0.5, 0.7, 0.9, 1.1]})
    print(f"ANGER dims & (valence<=v | hubert)-> {p5}  dev {fmt(d5)}\n{'':36s}test {fmt(t5)}")
    print(group_report(rows, r_w(**p5)))
    report['anger'] = {'hubert': (p1, d1, t1), 'dims': (p2, d2, t2), 'fusion': (p3, d3, t3),
                       'dims_valence': (p4, d4, t4), 'dims_valence_or_hubert': (p5, d5, t5)}

    # ---------------- violence: scream tag / dims gated by speech / fusion ----------------
    rows = rows_for('violence', [tagger, emo])
    r_s = lambda thr: (lambda f: f['scream_max'] >= thr)
    p1, d1, t1 = sweep(rows, r_s, {'thr': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]})
    print(f"\nVIOL  tagger scream_max>=thr       -> {p1}  dev {fmt(d1)}\n{'':36s}test {fmt(t1)}")
    r_g = lambda a, d, sp: (lambda f: f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d
                            and f['speech_max'] >= sp and f['speech_max'] > f['cry_max'])
    p2, d2, t2 = sweep(rows, r_g, {'a': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85], 'd': [0.55, 0.6, 0.65, 0.7, 0.75, 0.8], 'sp': [0.2, 0.3, 0.4, 0.5]})
    print(f"VIOL  dims gated by speech         -> {p2}  dev {fmt(d2)}\n{'':36s}test {fmt(t2)}")
    r_f = lambda thr, a, d, sp: (lambda f: f['scream_max'] >= thr or
                                 (f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d
                                  and f['speech_max'] >= sp and f['speech_max'] > f['cry_max']))
    p3, d3, t3 = sweep(rows, r_f, {'thr': [0.01, 0.02, 0.03, 0.05, 0.1, 1.1], 'a': [0.65, 0.7, 0.75, 0.8, 0.85], 'd': [0.6, 0.65, 0.7, 0.75, 0.8], 'sp': [0.2, 0.3, 0.4]})
    print(f"VIOL  scream OR gated dims         -> {p3}  dev {fmt(d3)}\n{'':36s}test {fmt(t3)}")
    print(group_report(rows, r_f(**p3)))
    r_fv = lambda thr, a, d, v: (lambda f: f['scream_max'] >= thr or
                                 (f.get('arousal_max', 0) >= a and f.get('dominance_max', 0) >= d
                                  and f.get('valence_mean', 1) <= v
                                  and f['speech_max'] >= 0.6 and f['speech_max'] > f['cry_max'] + 0.2))
    p4, d4, t4 = sweep(rows, r_fv, {'thr': [0.01, 0.02, 0.03, 0.05, 1.1], 'a': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 'd': [0.6, 0.65, 0.7, 0.75, 0.8], 'v': [0.3, 0.35, 0.4, 0.45, 1.0]})
    print(f"VIOL  scream OR dims&valence(gated)-> {p4}  dev {fmt(d4)}\n{'':36s}test {fmt(t4)}")
    print(group_report(rows, r_fv(**p4)))
    print("      scream threshold curve (test, scream only):")
    for thr in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        m = evaluate(rows, r_s(thr), 'test')
        print(f"        thr={thr:<5} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.4f}")
    report['violence'] = {'scream': (p1, d1, t1), 'gated_dims': (p2, d2, t2), 'fusion': (p3, d3, t3), 'fusion_valence': (p4, d4, t4)}

    out = ROOT / 'evaluation' / 'results' / 'fusion.json'
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
