import argparse
import time
import numpy as np

from .generators import set_precision, Z_raw, Z_o_placeholder
from .transforms import (
    fractional_part_min,
    forward_diff,
    logmellin_slope,
    mobius_twist,
    dirichlet_projection,
)
from .metrics import labels_for_range, auc_from_scores, polarity_index, split_windows, stability


def compute_G(start: int, end: int, use_zo: bool = False):
    """Compute Z(n) (and optional Z_o placeholder) for n in [start, end]."""
    if end < start:
        raise ValueError("end must be >= start")
    N = end - start + 1
    Z = np.zeros(N, dtype=float)
    Zo = np.zeros(N, dtype=float) if use_zo else None
    for i, n in enumerate(range(start, end + 1)):
        z = Z_raw(n)
        Z[i] = 0.0 if z is None else z
        if use_zo:
            zo = Z_o_placeholder(n)
            Zo[i] = 0.0 if zo is None else zo
    return Z, Zo


def feature_stack(Z, start, mods):
    feats = {
        "Z_raw": Z,
        "Frac_part_min": fractional_part_min(Z),
        "Forward_diff": forward_diff(Z),
        "LogMellin_slope": logmellin_slope(Z, start),
        "Mobius_twist": mobius_twist(Z, start),
    }
    for q in mods:
        feats[f"Dirichlet_proj_q={q}"] = dirichlet_projection(Z, start, q)
    return feats


def score_range(start, end, windows, window_size, use_zo, mods, dps):
    set_precision(dps)
    window_size = None if (window_size is None or window_size <= 0) else window_size
    ranges = split_windows(start, end, windows, window_size)
    results = []

    for (s, e) in ranges:
        labels = labels_for_range(s, e)
        Z, Zo = compute_G(s, e, use_zo=use_zo)
        feats = feature_stack(Z, s, mods)
        if use_zo:
            feats["Z_o_placeholder"] = Zo
        window_scores = {}
        for name, arr in feats.items():
            auc = auc_from_scores(arr, labels)
            pi = polarity_index(auc)
            window_scores[name] = (auc, pi)
        results.append(((s, e), window_scores))

    names = sorted({name for _, ws in results for name in ws.keys()})
    table = []
    for name in names:
        pis = []
        aucs = []
        for (_, ws) in results:
            if name in ws:
                auc, pi = ws[name]
                pis.append(pi)
                aucs.append(auc)
        if not pis:
            continue
        avg_auc = float(np.mean(aucs))
        avg_pi = float(np.mean(pis))
        is_stable = stability(pis, min_pi=0.2, max_rel_var=0.2)
        table.append((name, avg_auc, avg_pi, is_stable, pis))

    table.sort(key=lambda x: x[2], reverse=True)
    return ranges, table


def main():
    parser = argparse.ArgumentParser(description="Prime Polarity scorer for Z(p)/Z(o) generators.")
    parser.add_argument(
        "command",
        nargs="?",
        default="score",
        choices=["score"],
        help="Subcommand (only 'score' is supported; default).",
    )
    parser.add_argument("--start", type=int, default=100000)
    parser.add_argument("--end", type=int, default=120000)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=0, help="If 0, auto-derive from range/windows.")
    parser.add_argument("--use-zo", action="store_true", help="Include Z(o) placeholder feature.")
    parser.add_argument("--mods", type=str, default="4,5,8,12", help="Comma list of small moduli for Dirichlet projections.")
    parser.add_argument("--dps", type=int, default=50, help="mpmath precision digits.")
    args = parser.parse_args()

    if args.command != "score":
        parser.error(f"Unsupported command '{args.command}'. Only 'score' is available.")

    window_size = None if args.window_size == 0 else args.window_size
    mods = [int(x) for x in args.mods.split(",") if x.strip()]

    t0 = time.time()
    ranges, table = score_range(args.start, args.end, args.windows, window_size, args.use_zo, mods, args.dps)

    print("Windows:")
    for (s, e) in ranges:
        print(f"  [{s}, {e}]")

    print("\nResults (sorted by avg PI):")
    print(f"{'Feature':30s} {'AUC':>8s} {'PI':>8s}  {'Stable':>8s}  PIs-by-window")
    for name, avg_auc, avg_pi, is_stable, pis in table:
        stab = "*" if is_stable else ""
        print(f"{name:30s} {avg_auc:8.3f} {avg_pi:8.3f}  {stab:>8s}  {['%.3f' % p for p in pis]}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f}s with dps={args.dps}, mods={mods}")


if __name__ == '__main__':
    main()
