from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
        ticks: current number of ticks.
    """

    alpha = 2 * np.pi / resolution
    delta_ticks = ticks - prev_ticks
    dphi = alpha * delta_ticks

    return dphi, ticks


def estimate_pose(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x_curr:                  estimated x coordinate
        y_curr:                  estimated y coordinate
        theta_curr:              estimated heading
    """

    d_left = R * delta_phi_left
    d_right = R * delta_phi_right
    d_A = (d_left + d_right) / 2
    d_theta = (d_right - d_left) / baseline

    t_prev_pos = np.array([
        [np.cos(theta_prev), -np.sin(theta_prev), x_prev],
        [np.sin(theta_prev), np.cos(theta_prev), y_prev],
        [0, 0, 1]
    ])

    t_new_pos = ([
        [np.cos(d_theta), -np.sin(d_theta), d_A * np.cos(d_theta)],
        [np.sin(d_theta), np.cos(d_theta), d_A * np.sin(d_theta)],
        [0, 0, 1]
    ])

    new_pos = np.dot(t_prev_pos, t_new_pos)

    x_curr = new_pos[0, 2]
    y_curr = new_pos[1, 2]
    theta_curr = theta_prev + d_theta

    return x_curr, y_curr, theta_curr


if __name__ == "__main__":
    R = 0.0318
    baseline = 0.1
    x_prev = 0
    y_prev = 0
    theta_prev = 0
    delta_phi_right = 100
    delta_phi_left = 100

    print(estimate_pose(R, baseline, x_prev, y_prev, theta_prev, delta_phi_left, delta_phi_right))
