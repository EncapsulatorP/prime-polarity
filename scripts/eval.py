#!/usr/bin/env python3
import argparse
import json
import csv
import sys
import numpy as np
import pandas as pd


def auc_tie_aware(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    assert scores.shape[0] == labels.shape[0]
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    ranks = ranks[inv]
    pos = labels
    n_pos = int(pos.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def polarity_index(auc: float) -> float:
    return 2.0 * auc - 1.0


def main():
    ap = argparse.ArgumentParser(description="Evaluate features (AUC/PI) from a dataset CSV.")
    ap.add_argument("--data", required=True, help="CSV from scripts/make_dataset.py")
    ap.add_argument(
        "--features",
        default="Z_raw,FracPartMin,ForwardDiff,LogMellinSlope,MobiusFast,Dirichlet_q4,Dirichlet_q5,Dirichlet_q8,Dirichlet_q12,K3",
    )
    ap.add_argument("--format", choices=["txt", "md", "json", "csv"], default="md")
    ap.add_argument("--out", default="", help="Optional output path for json/csv/md")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = df["is_prime"].values.astype(bool)
    feats = [f.strip() for f in args.features.split(",") if f.strip()]

    rows = []
    for f in feats:
        if f not in df.columns:
            continue
        x = df[f].values.astype(float)
        auc = auc_tie_aware(x, y)
        pi = polarity_index(auc)
        rows.append({"feature": f, "auc": round(auc, 6), "pi": round(pi, 6)})
    rows.sort(key=lambda r: r["pi"], reverse=True)

    if args.format == "txt":
        print(f"{'feature':30s} {'auc':>8s} {'pi':>8s}")
        for r in rows:
            print(f"{r['feature']:30s} {r['auc']:8.3f} {r['pi']:8.3f}")
        return

    if args.format == "md":
        out = ["| feature | auc | pi |", "|---|---:|---:|"]
        for r in rows:
            out.append(f"| {r['feature']} | {r['auc']:.3f} | {r['pi']:.3f} |")
        s = "\n".join(out)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(s + "\n")
            print(f"Wrote metrics table → {args.out}")
        else:
            print(s)
        return

    if args.format == "json":
        payload = {"results": rows}
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote JSON → {args.out}")
        else:
            print(json.dumps(payload, indent=2))
        return

    if args.format == "csv":
        target = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
        with target as f:
            w = csv.DictWriter(f, fieldnames=["feature", "auc", "pi"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        if args.out and target is not sys.stdout:
            print(f"Wrote CSV → {args.out}")


if __name__ == "__main__":
    main()
