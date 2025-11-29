import numpy as np
from .sieves import prime_sieve_up_to

def labels_for_range(start: int, end: int) -> np.ndarray:
    """Boolean labels array for n in [start, end], True if prime."""
    is_prime = prime_sieve_up_to(end)
    labels = np.array([is_prime[n] for n in range(start, end+1)], dtype=bool)
    return labels

def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute ROC AUC (tie-aware) without sklearn using average ranks on ties.
    """
    assert len(scores) == len(labels)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=float)

    # Average ranks for tied scores
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j

    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    ranks = ranks[inv]

    pos = labels
    neg = ~labels
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg)
    return float(auc)

def polarity_index(auc: float) -> float:
    return 2.0*auc - 1.0

def split_windows(start: int, end: int, windows: int, window_size: int = None):
    """Create [start_i, end_i] windows covering [start,end]."""
    if window_size is None:
        N = end - start + 1
        window_size = max(10, N // windows)
    ranges = []
    s = start
    for _ in range(windows):
        e = min(end, s + window_size - 1)
        ranges.append((s, e))
        s = e + 1
        if s > end:
            break
    return ranges

def stability(pis, min_pi: float = 0.2, max_rel_var: float = 0.2) -> bool:
    """
    A crude stability test: all PI >= min_pi and relative std dev <= max_rel_var.
    """
    arr = np.array(pis, dtype=float)
    if (arr < min_pi).any():
        return False
    m = arr.mean()
    if m == 0:
        return False
    rel_std = arr.std() / abs(m)
    return rel_std <= max_rel_var
