"""Test CLI functionality."""

from prime_polarity.cli import compute_G, feature_stack, score_range
from prime_polarity.generators import set_precision


def test_compute_g_basic():
    """Test basic generator computation."""
    set_precision(50)
    Z, Zo = compute_G(3, 5, use_zo=False)

    assert len(Z) == 3  # 3, 4, 5
    assert Zo is None
    assert all(z > 0 for z in Z)  # Z(n) should be positive


def test_compute_g_with_zo():
    """Test generator computation with Z(o) values."""
    set_precision(50)
    Z, Zo = compute_G(3, 5, use_zo=True)

    assert len(Z) == len(Zo) == 3
    assert Zo is not None


def test_feature_stack():
    """Test feature transformation stack."""
    set_precision(50)
    Z, _ = compute_G(10, 15, use_zo=False)
    mods = [4, 5]

    features = feature_stack(Z, 10, mods)

    expected_keys = [
        "Z_raw",
        "Frac_part_min",
        "Forward_diff",
        "LogMellin_slope",
        "Mobius_twist",
        "Dirichlet_proj_q=4",
        "Dirichlet_proj_q=5",
    ]

    for key in expected_keys:
        assert key in features
        assert len(features[key]) == len(Z)


def test_score_range_small():
    """Test scoring on a small range."""
    set_precision(30)  # Lower precision for faster test

    ranges, table = score_range(
        start=100, end=150, windows=2, window_size=25, use_zo=False, mods=[4], dps=30
    )

    assert len(ranges) == 2
    assert len(table) > 0  # Should have some feature results

    # Check table structure
    for entry in table:
        name, avg_auc, avg_pi, is_stable, pis = entry
        assert isinstance(name, str)
        assert isinstance(avg_auc, float)
        assert isinstance(avg_pi, float)
        assert isinstance(is_stable, bool)
        assert isinstance(pis, list)
