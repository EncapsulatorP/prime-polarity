# Prime Polarity

Prime Polarity scores how well numeric generators separate primes from composites. It computes ROC AUC and the Polarity Index (PI = 2 * AUC - 1) for raw generators and for a set of simple transforms (differences, Möbius twist, fractional-part proximity, Dirichlet projections).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Installation
```bash
pip install prime-polarity
```

## Quick CLI run
```bash
prime-polarity score --start 100000 --end 120000 --windows 3 --window-size 5000 --mods 4,5,8,12
```
Outputs per-window and average AUC/PI for each feature. Use `--use-zo` to include the placeholder Z(o) projection and `--dps` to set mpmath precision.

## What is scored
- `Z_raw(n) = exp(pi * zeta(n-1) / n) + 1` (near-constant baseline).
- Transforms: fractional-part minimum, forward difference, log-Mellin slope, Möbius twist, Dirichlet character projections for small moduli.
- Labels: primes vs composites over `[start, end]` via a sieve; PI is neutral near 0, positive for direct signals, negative for reverse signals.

## Python API
```python
from prime_polarity.cli import score_range

ranges, table = score_range(
    start=10_000,
    end=12_000,
    windows=3,
    window_size=2_000,
    use_zo=False,
    mods=[4, 5, 8, 12],
    dps=50,
)
```
`table` contains `(feature, avg_auc, avg_pi, is_stable, pis_by_window)`.

## Dataset + offline evaluation
```bash
python scripts/make_dataset.py --start 100000 --end 120000 --out data/polarity.csv
python scripts/eval.py --data data/polarity.csv --format md --out results.md
```

## Development
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check .
black .
pytest
```

## Limitations and cautions
- Tie handling matters: use tie-aware AUC (implemented in `metrics.py` and `scripts/eval.py`) for plateaued features.
- Leakage: parity and small-modulus artifacts can inflate PI; prefer odd-only or coprime masks when claiming signal.
- Multiple testing: confirm stability across disjoint windows; compare against null baselines.
- Performance: large ranges with high precision are slow; start with modest windows to explore.

## License
MIT
