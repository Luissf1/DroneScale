"""
Quadrotor Dynamics Model
========================

Implements the 6-DOF rigid body dynamics of a quadrotor UAV using
Newton-Euler formalism. Supports parameterized quadrotor variants
for scaling analysis.

State vector: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QuadrotorParameters:
    """Physical parameters of a quadrotor."""

    mass: float = 1.0              # kg
    arm_length: float = 0.25       # m
    Ixx: float = 0.1               # kg·m²
    Iyy: float = 0.1               # kg·m²
    Izz: float = 0.2               # kg·m²
    gravity: float = 9.81          # m/s²
    thrust_coeff: float = 3.13e-5  # N·s²
    drag_coeff: float = 7.5e-7     # N·m·s²
    friction: float = 0.1          # N·s/m


class QuadrotorDynamics:
    """
    Quadrotor dynamics model with PID control interface.

    Attributes:
        params: QuadrotorParameters instance.
        g: gravitational acceleration (m/s²).
    """

    def __init__(self, params: QuadrotorParameters):
        self.params = params
        self.g = params.gravity

    # ------------------------------------------------------------------ #
    # Control effort computation
    # ------------------------------------------------------------------ #
    def compute_control_effort(
        self,
        state: np.ndarray,
        gains: np.ndarray,
        reference: np.ndarray,
        integral_errors: np.ndarray = None,
        dt: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PID control efforts for altitude and attitude.

        Args:
            state: Current state vector [12].
            gains: PID gains [Kp_z, Ki_z, Kd_z, Kp_phi, ..., Kd_psi] [12].
            reference: Desired state [z, phi, theta, psi] [4].
            integral_errors: Accumulated integral errors [4] (optional).
            dt: Time step for integral update.

        Returns:
            controls: [U1, U2, U3, U4].
            errors: [e_z, e_phi, e_theta, e_psi].
            integral_errors: Updated integral error accumulator [4].
        """
        _, _, z, phi, theta, psi, _, _, vz, p, q, r = state
        z_des, phi_des, theta_des, psi_des = reference

        # Tracking errors
        e_z = z_des - z
        e_phi = phi_des - phi
        e_theta = theta_des - theta
        e_psi = psi_des - psi
        errors = np.array([e_z, e_phi, e_theta, e_psi])

        # Error derivatives
        de = np.array([-vz, -p, -q, -r])

        # Update integral errors
        if integral_errors is None:
            integral_errors = np.zeros(4)
        if dt > 0.0:
            integral_errors = integral_errors + errors * dt

        # Extract gains per axis
        Kp = np.array([gains[0], gains[3], gains[6], gains[9]])
        Ki = np.array([gains[1], gains[4], gains[7], gains[10]])
        Kd = np.array([gains[2], gains[5], gains[8], gains[11]])

        # PID law
        U = Kp * errors + Ki * integral_errors + Kd * de

        # Saturation limits
        U_max = self.params.mass * self.g * 2.0
        U1 = float(np.clip(U[0], 0.0, U_max))
        U2 = float(np.clip(U[1], -3.0, 3.0))
        U3 = float(np.clip(U[2], -3.0, 3.0))
        U4 = float(np.clip(U[3], -1.0, 1.0))

        return np.array([U1, U2, U3, U4]), errors, integral_errors

    # ------------------------------------------------------------------ #
    # ODE derivatives
    # ------------------------------------------------------------------ #
    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """
        Compute state derivatives for ODE solver.

        Args:
            t: Current time (unused, kept for solver signature).
            state: State vector [12].
            controls: Control inputs [U1, U2, U3, U4].

        Returns:
            dstate: State derivatives [12].
        """
        _, _, _, phi, theta, psi, vx, vy, vz, p, q, r = state
        U1, U2, U3, U4 = controls

        m = self.params.mass
        g = self.g
        l = self.params.arm_length
        Kf = self.params.friction

        Ixx, Iyy, Izz = self.params.Ixx, self.params.Iyy, self.params.Izz

        # Trigonometry
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # Forces in body frame
        Fx = (spsi * sphi + cpsi * stheta * cphi) * U1
        Fy = (-cpsi * sphi + spsi * stheta * cphi) * U1
        Fz = (ctheta * cphi) * U1

        # Linear accelerations
        ax = Fx / m - Kf * vx
        ay = Fy / m - Kf * vy
        az = Fz / m - g - Kf * vz

        # Angular accelerations
        p_dot = (l * U2 + q * r * (Iyy - Izz)) / Ixx
        q_dot = (l * U3 + p * r * (Izz - Ixx)) / Iyy
        r_dot = (U4 + p * q * (Ixx - Iyy)) / Izz

        return np.array([
            vx, vy, vz, p, q, r,
            ax, ay, az, p_dot, q_dot, r_dot,
        ])

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def make_scaled_copy(self, mass_factor: float) -> "QuadrotorDynamics":
        """
        Return a new QuadrotorDynamics object scaled by geometric similarity.

        Arm length scales as mass^(1/3); inertia scales as mass * arm².
        """
        m = self.params.mass * mass_factor
        l = self.params.arm_length * (mass_factor ** (1.0 / 3.0))
        inertia_scale = mass_factor * (mass_factor ** (2.0 / 3.0))

        new_params = QuadrotorParameters(
            mass=m,
            arm_length=l,
            Ixx=self.params.Ixx * inertia_scale,
            Iyy=self.params.Iyy * inertia_scale,
            Izz=self.params.Izz * inertia_scale,
            gravity=self.params.gravity,
            thrust_coeff=self.params.thrust_coeff,
            drag_coeff=self.params.drag_coeff,
            friction=self.params.friction,
        )
        return QuadrotorDynamics(new_params)