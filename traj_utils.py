"""
traj_utils.py - Trajectory generators and utilities for BJORK.
Generates initial 2D non-Cartesian sampling trajectories (Radial, Spiral, Golden-Angle)
and provides visualization tools.
"""

import math
import numpy as np
import torch


def generate_radial_trajectory(num_shots=16, nfe=512, golden_angle=True):
    """
    Generate a 2D radial spokes k-space trajectory.

    Args:
        num_shots (int): Number of radial spokes.
        nfe (int): Number of frequency encoding (readout) points per spoke.
        golden_angle (bool): If True, use golden angle spacing (~111.246 deg).
                             If False, use uniform spacing in [0, pi).

    Returns:
        traj (np.ndarray): Shape (2, num_shots, nfe) with values normalized to [-pi, pi].
    """
    traj = np.zeros((2, num_shots, nfe), dtype=np.float32)
    radii = np.linspace(-np.pi, np.pi, nfe, endpoint=True)

    if golden_angle:
        phi = (np.sqrt(5.0) - 1.0) / 2.0  # Golden ratio conjugate
        ga_rad = np.pi * phi              # ~111.246 degrees
        angles = np.array([(i * ga_rad) % np.pi for i in range(num_shots)])
    else:
        angles = np.linspace(0, np.pi, num_shots, endpoint=False)

    for i, angle in enumerate(angles):
        traj[0, i, :] = radii * np.cos(angle)
        traj[1, i, :] = radii * np.sin(angle)

    return traj


def generate_spiral_trajectory(num_shots=16, nfe=512, alpha=1.0):
    """
    Generate an Archimedean multi-arm spiral trajectory.

    Args:
        num_shots (int): Number of interleaves.
        nfe (int): Readout points per interleave.
        alpha (float): Power parameter for Archimedean spiral radius.

    Returns:
        traj (np.ndarray): Shape (2, num_shots, nfe) in [-pi, pi].
    """
    traj = np.zeros((2, num_shots, nfe), dtype=np.float32)
    tau = np.linspace(0, 1, nfe)
    radii = np.pi * (tau ** alpha)

    base_angle = 2 * np.pi * 3.0 * tau  # 3 full turns
    interleave_angle = 2 * np.pi / num_shots

    for i in range(num_shots):
        angle = base_angle + i * interleave_angle
        traj[0, i, :] = radii * np.cos(angle)
        traj[1, i, :] = radii * np.sin(angle)

    return traj


def save_initial_trajectory(path="radial_init.npy", traj_type="radial", num_shots=16, nfe=512):
    """
    Generate and save initial trajectory to a .npy file.
    """
    if traj_type == "radial":
        traj = generate_radial_trajectory(num_shots=num_shots, nfe=nfe, golden_angle=True)
    elif traj_type == "spiral":
        traj = generate_spiral_trajectory(num_shots=num_shots, nfe=nfe)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    np.save(path, traj)
    print(f"Saved initial {traj_type} trajectory with shape {traj.shape} to {path}")
    return traj


if __name__ == "__main__":
    traj = save_initial_trajectory("radial_init.npy", num_shots=16, nfe=512)
    print("Trajectory stats: min =", traj.min(), "max =", traj.max(), "shape =", traj.shape)
