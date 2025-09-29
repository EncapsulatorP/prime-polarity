import numpy as np
from prime_polarity.generators import set_precision, Z_raw
from prime_polarity.sieves import prime_sieve_up_to, mobius_sieve_up_to
from prime_polarity.metrics import auc_from_scores, polarity_index


def test_z_raw_basic():
    set_precision(50)
    z3 = Z_raw(3)
    assert z3 is not None and z3 > 0


def test_prime_sieve_small():
    is_prime = prime_sieve_up_to(30)
    primes = [i for i, b in enumerate(is_prime) if b]
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


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
