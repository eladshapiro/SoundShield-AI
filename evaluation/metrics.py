"""Binary-classification metrics for the evaluation suite."""
from typing import Dict, List, Sequence

import numpy as np


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                   y_score: Sequence[float] = None) -> Dict[str, float]:
    """Precision / recall / F1 / FPR (+ ROC-AUC and AP when scores are given)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    out = {
        'n_pos': int(tp + fn), 'n_neg': int(fp + tn),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr,
        'accuracy': (tp + tn) / max(1, len(yt)),
    }
    if y_score is not None and len(set(yt.tolist())) == 2:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            ys = np.asarray(y_score, dtype=float)
            out['roc_auc'] = float(roc_auc_score(yt, ys))
            out['avg_precision'] = float(average_precision_score(yt, ys))
        except Exception:  # pragma: no cover - sklearn missing or degenerate input
            pass
    return out


def group_rates(groups: Sequence[str], y_true: Sequence[int],
                y_pred: Sequence[int]) -> Dict[str, Dict[str, float]]:
    """Per-group hit rate: recall for positive groups, FPR for negative groups."""
    out: Dict[str, Dict[str, float]] = {}
    groups = list(groups)
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    for g in sorted(set(groups)):
        idx = np.array([i for i, gg in enumerate(groups) if gg == g])
        t, p = yt[idx], yp[idx]
        label = int(round(t.mean())) if len(t) else 0
        rate = float(p.mean()) if len(p) else 0.0
        out[g] = {'n': int(len(idx)), 'label': label,
                  'positive_rate': rate,
                  'recall' if label == 1 else 'fpr': rate}
    return out


def best_threshold(y_true: Sequence[int], y_score: Sequence[float],
                   min_precision: float = 0.0) -> Dict[str, float]:
    """Sweep thresholds on a score and return the F1-optimal operating point."""
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    best = {'threshold': 0.5, 'f1': -1.0}
    for thr in np.unique(ys):
        yp = (ys >= thr).astype(int)
        m = binary_metrics(yt, yp)
        if m['precision'] >= min_precision and m['f1'] > best['f1']:
            best = {'threshold': float(thr), **{k: m[k] for k in ('precision', 'recall', 'f1', 'fpr')}}
    return best
