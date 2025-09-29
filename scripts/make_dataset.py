#!/usr/bin/env python3
import argparse
import math
import os
import numpy as np
import pandas as pd
from mpmath import mp, zeta, pi


def prime_sieve_up_to(n: int):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : n + 1 : step] = [False] * (((n - start) // step) + 1)
        p += 1
    return is_prime


def mobius_sieve_up_to(n: int):
    mu = [1] * (n + 1)
    is_prime = [True] * (n + 1)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        j = 0
        while j < len(primes) and i * primes[j] <= n:
            p = primes[j]
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
            j += 1
    mu[0] = 0
    return mu


def Z_raw(n: int):
    if n <= 2:
        return float("nan")
    return float(mp.e ** (pi * zeta(n - 1) / n) + 1)


def fractional_part_min(x: np.ndarray) -> np.ndarray:
    frac = x - np.floor(x)
    return np.minimum(frac, 1.0 - frac)


def forward_diff(x: np.ndarray) -> np.ndarray:
    d = np.empty_like(x)
    d[:-1] = x[1:] - x[:-1]
    d[-1] = d[-2] if len(x) >= 2 else 0.0
    return d


def logmellin_slope(x: np.ndarray, n0: int) -> np.ndarray:
    n = np.arange(n0, n0 + len(x), dtype=float)
    d = forward_diff(x)
    return n * d


def mobius_twist_fast(G_values: np.ndarray, n0: int) -> np.ndarray:
    N = len(G_values)
    max_n = n0 + N - 1
    mu = mobius_sieve_up_to(max_n)
    out = np.zeros(N, dtype=float)
    for d in range(1, max_n + 1):
        mu_d = mu[d]
        if mu_d == 0:
            continue
        start_n = ((n0 + d - 1) // d) * d
        for n in range(start_n, max_n + 1, d):
            q = n // d
            q_idx = q - n0
            n_idx = n - n0
            if 0 <= q_idx < N and 0 <= n_idx < N:
                out[n_idx] += mu_d * G_values[q_idx]
    return out


def dirichlet_projection(G_values: np.ndarray, n0: int, modulus: int) -> np.ndarray:
    N = len(G_values)
    n = np.arange(n0, n0 + N)

    def chi_mod4(m):
        if m % 2 == 0:
            return 0
        return 1 if (m % 4) == 1 else -1

    def chi_mod5(m):
        r = m % 5
        if r == 0:
            return 0
        return 1 if r in (1, 4) else -1

    def chi_mod8(m):
        if math.gcd(m, 8) != 1:
            return 0
        r = m % 8
        return 1 if r in (1, 7) else -1

    def chi_mod12(m):
        if math.gcd(m, 12) != 1:
            return 0
        r = m % 12
        return 1 if r in (1, 11) else -1

    if modulus == 4:
        chi = np.array([chi_mod4(int(x)) for x in n], dtype=float)
    elif modulus == 5:
        chi = np.array([chi_mod5(int(x)) for x in n], dtype=float)
    elif modulus == 8:
        chi = np.array([chi_mod8(int(x)) for x in n], dtype=float)
    elif modulus == 12:
        chi = np.array([chi_mod12(int(x)) for x in n], dtype=float)
    else:
        raise ValueError("Unsupported modulus. Use one of {4,5,8,12}.")
    return chi * G_values


def k3_golden_ln_bands(n0: int, N: int) -> np.ndarray:
    PHI = (1 + 5**0.5) / 2.0
    n = np.arange(n0, n0 + N, dtype=float)
    n = np.maximum(n, 2.0)
    t = (np.log(n) / np.log(PHI)) % 1.0
    bands = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
    dists = np.min(np.abs(t[:, None] - bands[None, :]), axis=1)
    dists = np.minimum(dists, 1.0 - dists)
    score = 1.0 - 3.0 * dists
    return np.clip(score, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dps", type=int, default=50, help="mpmath precision")
    args = ap.parse_args()

    mp.dps = max(30, int(args.dps))

    start, end = args.start, args.end
    n = np.arange(start, end + 1)
    is_prime = prime_sieve_up_to(end)
    labels = np.array([is_prime[i] for i in n], dtype=bool)

    # Z_raw
    Z = np.array([Z_raw(int(i)) for i in n], dtype=float)
    # Features
    f_frac = fractional_part_min(Z.copy())
    f_diff = forward_diff(Z.copy())
    f_logm = logmellin_slope(Z.copy(), start)
    f_mobi = mobius_twist_fast(Z.copy(), start)
    f_q4 = dirichlet_projection(Z.copy(), start, 4)
    f_q5 = dirichlet_projection(Z.copy(), start, 5)
    f_q8 = dirichlet_projection(Z.copy(), start, 8)
    f_q12 = dirichlet_projection(Z.copy(), start, 12)
    f_k3 = k3_golden_ln_bands(start, len(Z))

    df = pd.DataFrame(
        {
            "n": n,
            "is_prime": labels,
            "Z_raw": Z,
            "FracPartMin": f_frac,
            "ForwardDiff": f_diff,
            "LogMellinSlope": f_logm,
            "MobiusFast": f_mobi,
            "Dirichlet_q4": f_q4,
            "Dirichlet_q5": f_q5,
            "Dirichlet_q8": f_q8,
            "Dirichlet_q12": f_q12,
            "K3": f_k3,
        }
    )

    outp = args.out
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"Wrote dataset: {outp}  rows={len(df)}")


if __name__ == "__main__":
    main()
