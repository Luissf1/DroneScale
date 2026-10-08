"""
PID Controller
==============

Discrete-time PID controller with anti-windup and output saturation.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PIDGains:
    """PID gains for a single axis."""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0


@dataclass
class PIDController:
    """
    Single-axis PID controller with anti-windup.

    Attributes:
        gains: PIDGains instance.
        output_limits: (min, max) tuple; None disables saturation.
        integral_limit: Optional limit on the integral term.
    """

    gains: PIDGains = field(default_factory=PIDGains)
    output_limits: Optional[tuple] = None
    integral_limit: Optional[float] = None

    _integral: float = 0.0
    _prev_error: float = 0.0

    def reset(self) -> None:
        """Reset internal integrator and derivative memory."""
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Compute PID output for a given error.

        Args:
            error: Reference minus measurement.
            dt: Time step since last update (seconds).

        Returns:
            Control output.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        # Proportional
        P = self.gains.Kp * error

        # Integral
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = float(
                np.clip(self._integral, -self.integral_limit, self.integral_limit)
            )
        I = self.gains.Ki * self._integral

        # Derivative
        D = self.gains.Kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = P + I + D

        # Anti-windup via conditional integration
        if self.output_limits is not None:
            lo, hi = self.output_limits
            clipped = float(np.clip(output, lo, hi))
            if clipped != output:
                # Back off the integrator if saturated
                self._integral -= error * dt
            output = clipped

        return output