"""Tests for quadrotor dynamics."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.quadrotor_model import QuadrotorDynamics, QuadrotorParameters


def test_default_parameters_are_positive():
    params = QuadrotorParameters()
    assert params.mass > 0
    assert params.arm_length > 0
    assert params.Ixx > 0 and params.Iyy > 0 and params.Izz > 0


def test_derivatives_return_correct_shape():
    params = QuadrotorParameters()
    dyn = QuadrotorDynamics(params)
    state = np.zeros(12)
    controls = np.array([params.mass * params.gravity, 0.0, 0.0, 0.0])
    dstate = dyn.derivatives(0.0, state, controls)
    assert dstate.shape == (12,)


def test_hover_derivative_has_no_vertical_acceleration():
    """At hover with proper U1, vertical acceleration should be ~0."""
    params = QuadrotorParameters()
    dyn = QuadrotorDynamics(params)
    state = np.zeros(12)
    controls = np.array([params.mass * params.gravity, 0.0, 0.0, 0.0])
    dstate = dyn.derivatives(0.0, state, controls)
    # dstate[8] is az
    assert abs(dstate[8]) < 1e-6


def test_scaled_copy_preserves_density():
    params = QuadrotorParameters()
    dyn = QuadrotorDynamics(params)
    dyn2 = dyn.make_scaled_copy(8.0)
    # Arm length scales as m^(1/3) = 2
    assert abs(dyn2.params.arm_length - 2.0 * params.arm_length) < 1e-9
    assert abs(dyn2.params.mass - 8.0) < 1e-9


def test_control_effort_clipping():
    params = QuadrotorParameters()
    dyn = QuadrotorDynamics(params)
    # Enormous gains → must clip
    gains = np.full(12, 1e6)
    state = np.zeros(12)
    reference = np.array([1000.0, 10.0, 10.0, 10.0])
    controls, _, _ = dyn.compute_control_effort(state, gains, reference)
    assert controls[1] <= 3.0 and controls[1] >= -3.0
    assert controls[3] <= 1.0 and controls[3] >= -1.0