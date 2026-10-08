"""Tests for the fuzzy adaptive scaler."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.fuzzy_adaptive import (
    FuzzyAdaptiveScaler,
    GaussianMF,
    TriangularMF,
)


def test_gaussian_mf_peaks_at_mean():
    mf = GaussianMF(mean=5.0, sigma=1.0)
    assert abs(mf(np.array([5.0]))[0] - 1.0) < 1e-9


def test_triangular_mf_shape():
    mf = TriangularMF(0.0, 1.0, 2.0)
    assert mf(np.array([1.0]))[0] == 1.0
    assert mf(np.array([0.0]))[0] == 0.0
    assert mf(np.array([2.0]))[0] == 0.0


def test_fuzzy_scaler_outputs_in_range():
    scaler = FuzzyAdaptiveScaler()
    alpha_p, alpha_i, alpha_d = scaler.compute_corrections(
        payload_variation=15.0,
        turbulence_level=0.4,
        battery_state=0.7,
        error_dynamics=0.3,
    )
    for a in (alpha_p, alpha_i, alpha_d):
        assert 0.5 <= a <= 2.0


def test_fuzzy_scaler_nominal_correction_close_to_one():
    scaler = FuzzyAdaptiveScaler()
    alpha_p, alpha_i, alpha_d = scaler.compute_corrections(
        payload_variation=0.0,
        turbulence_level=0.1,
        battery_state=1.0,
        error_dynamics=0.05,
    )
    # Under nominal conditions, corrections should be moderate
    for a in (alpha_p, alpha_i, alpha_d):
        assert 0.5 <= a <= 2.0


def test_fuzzy_scaler_responds_to_heavy_payload():
    scaler = FuzzyAdaptiveScaler()
    alpha_p_light, _, _ = scaler.compute_corrections(0.0, 0.1, 1.0, 0.1)
    alpha_p_heavy, _, _ = scaler.compute_corrections(35.0, 0.1, 1.0, 0.1)
    # Heavy payload should require larger proportional correction
    assert alpha_p_heavy > alpha_p_light


def test_fuzzy_scaler_apply_to_gains_shape():
    scaler = FuzzyAdaptiveScaler()
    gains = np.ones(12)
    out = scaler.apply_to_gains(gains, payload_variation=10.0)
    assert out.shape == (12,)
    assert np.all(out > 0)