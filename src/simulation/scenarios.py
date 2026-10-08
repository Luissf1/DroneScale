"""
Flight Scenarios
================

Defines the flight scenarios (E1–E5) used in scaling analysis and
disturbance experiments, along with disturbance parameters.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Scenario:
    """A flight scenario with a reference command and optional disturbances."""

    name: str
    code: str
    reference: np.ndarray           # [z, phi, theta, psi]
    payload_pct: float = 0.0        # extra payload (%)
    turbulence: float = 0.0         # 0–1
    battery: float = 1.0            # 0–1


def get_default_scenarios() -> List[Scenario]:
    """Return the five canonical scaling-analysis scenarios."""
    return [
        Scenario(
            name="Stationary Takeoff",
            code="E1",
            reference=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        Scenario(
            name="Inclined Takeoff",
            code="E2",
            reference=np.array([1.5, 0.1, -0.1, 0.0]),
        ),
        Scenario(
            name="Transitional Climb",
            code="E3",
            reference=np.array([2.0, -0.2, 0.2, 0.0]),
        ),
        Scenario(
            name="Yaw-Controlled Takeoff",
            code="E4",
            reference=np.array([1.0, 0.0, 0.0, np.pi / 4]),
        ),
        Scenario(
            name="Multi-Axis Maneuver",
            code="E5",
            reference=np.array([0.5, -0.1, -0.1, -np.pi / 6]),
        ),
    ]


def get_disturbance_scenarios() -> List[Scenario]:
    """Return disturbance-focused scenarios for fuzzy adaptive testing."""
    base = get_default_scenarios()[0]  # Use E1 reference
    return [
        Scenario("Nominal", "D0", base.reference, payload_pct=0, turbulence=0.0, battery=1.0),
        Scenario("Payload +15%", "D1", base.reference, payload_pct=15, turbulence=0.0, battery=1.0),
        Scenario("Payload +30%", "D2", base.reference, payload_pct=30, turbulence=0.0, battery=1.0),
        Scenario("Moderate Wind", "D3", base.reference, payload_pct=0, turbulence=0.4, battery=1.0),
        Scenario("Strong Wind", "D4", base.reference, payload_pct=0, turbulence=0.8, battery=1.0),
        Scenario("Battery 50%", "D5", base.reference, payload_pct=0, turbulence=0.0, battery=0.5),
        Scenario("Battery 20%", "D6", base.reference, payload_pct=0, turbulence=0.0, battery=0.2),
        Scenario("Combined", "D7", base.reference, payload_pct=25, turbulence=0.6, battery=0.3),
    ]