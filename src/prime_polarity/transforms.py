import math
import numpy as np
from .sieves import mobius_sieve_up_to, divisors_up_to


def fractional_part_min(x: np.ndarray) -> np.ndarray:
    """Elementwise s = min(frac(x), 1-frac(x))."""
    frac = x - np.floor(x)
    return np.minimum(frac, 1.0 - frac)


def forward_diff(x: np.ndarray) -> np.ndarray:
    """Compute forward difference, last element repeated to keep same length."""
    d = np.empty_like(x)
    d[:-1] = x[1:] - x[:-1]
    d[-1] = d[-2] if len(x) >= 2 else 0.0
    return d


def logmellin_slope(x: np.ndarray, n0: int) -> np.ndarray:
    """
    Approximate Δ_log G(n) = n*(G(n+1)-G(n)); n0 is the starting n for x[0].
    """
    n = np.arange(n0, n0 + len(x), dtype=float)
    d = forward_diff(x)
    return n * d


def mobius_twist(G_values: np.ndarray, n0: int) -> np.ndarray:
    """
    Compute M[G](n) = sum_{d|n} mu(d) * G(n/d) for n in [n0, n0+len-1].
    """
    N = len(G_values)
    max_n = n0 + N - 1
    mu = mobius_sieve_up_to(max_n)
    divs = divisors_up_to(max_n)

    out = np.zeros(N, dtype=float)
    for idx, n in enumerate(range(n0, n0 + N)):
        acc = 0.0
        for d in divs[n]:
            mu_d = mu[d]
            if mu_d == 0:
                continue
            q = n // d
            q_idx = q - n0
            if 0 <= q_idx < N:
                acc += mu_d * G_values[q_idx]
        out[idx] = acc
    return out


def dirichlet_projection(
    G_values: np.ndarray, n0: int, modulus: int, kind: str = "odd"
) -> np.ndarray:
    """
    Simple real Dirichlet character projection for small moduli.
    Supported moduli: 4, 5, 8, 12 (demo characters).
    """
    N = len(G_values)
    n = np.arange(n0, n0 + N)

    def chi_mod4(m):
        if m % 2 == 0:
            return 0
        return 1 if (m % 4) == 1 else -1  # primitive odd real character

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
