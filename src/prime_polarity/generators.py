from typing import Optional
from mpmath import mp, zeta, pi


def set_precision(dps: int = 50) -> None:
    """Set mpmath decimal precision (digits)."""
    mp.dps = max(30, int(dps))


def Z_raw(n: int) -> Optional[float]:
    """Z(n) = exp(pi * zeta(n-1) / n) + 1. Defined for n>=3 (zeta(1) pole)."""
    if n <= 2:
        return None
    val = mp.e ** (pi * zeta(n - 1) / n) + 1
    return float(val)


def Z_o_placeholder(n: int) -> Optional[float]:
    """
    Placeholder for Z(o): a Dirichlet-character flavored variant.
    Example: project Z(n) through a real character mod 4.
    """
    z = Z_raw(n)
    if z is None:
        return None
    # χ_4: 0 if even; +1 if n≡1 (mod 4); -1 if n≡3 (mod 4)
    r = n & 3
    if (n % 2) == 0:
        chi = 0.0
    elif r == 1:
        chi = 1.0
    else:  # r==3
        chi = -1.0
    return float(chi * z)
