import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from prime_polarity.generators import set_precision, Z_raw
from prime_polarity.sieves import prime_sieve_up_to, mobius_sieve_up_to
from prime_polarity.metrics import auc_from_scores, polarity_index
from prime_polarity.cli import score_range

def test_z_raw_basic():
    set_precision(50)
    z3 = Z_raw(3)
    assert z3 is not None and z3 > 0

def test_prime_sieve_small():
    is_prime = prime_sieve_up_to(30)
    primes = [i for i,b in enumerate(is_prime) if b]
    assert primes == [2,3,5,7,11,13,17,19,23,29]

def test_mobius_sieve_small():
    mu = mobius_sieve_up_to(10)
    assert mu[1] == 1 and mu[2] == -1 and mu[3] == -1 and mu[4] == 0 and mu[6] == 1

def test_auc_sanity():
    labels = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    auc = auc_from_scores(scores, labels)
    assert abs(auc - 1.0) < 1e-9
    pi = polarity_index(auc)
    assert abs(pi - 1.0) < 1e-9

def test_auc_tie_handling():
    labels = np.array([True, False, True, False])
    scores = np.array([0.5, 0.5, 0.9, 0.1])
    # With ties at 0.5, tie-aware AUC should be 0.875
    auc = auc_from_scores(scores, labels)
    assert abs(auc - 0.875) < 1e-9

def test_score_range_runs():
    ranges, table = score_range(start=10, end=30, windows=2, window_size=0, use_zo=False, mods=[4], dps=30)
    assert ranges
    assert table
    names = [row[0] for row in table]
    assert "Z_raw" in names
