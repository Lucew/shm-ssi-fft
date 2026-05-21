"""
Simulate bridge-like acceleration data
This was generated using ChatGPT as it is way outside of my expertise, no guarantees that it works
"""
import numpy as np
import scipy.linalg as splg
import scipy.signal as sps


def make_bridge_like_mode_shapes(n_sensors, n_modes):
    """
    Create simple sine-like mode shapes along a 1D bridge span.
    This is only for simulation.
    """
    x = np.linspace(0.08, 0.92, n_sensors)

    Phi = np.column_stack([
        np.sin((r + 1) * np.pi * x)
        for r in range(n_modes)
    ])

    # Normalize each mode shape
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

    return Phi, x


def simulate_acceleration_dataset(
    fs=100.0,
    duration=300.0,
    freqs_hz=(2.0, 5.0, 8.0),
    damping=(0.01, 0.015, 0.02),
    n_sensors=6,
    excitation_std=1.0,
    noise_std=0.02,
    burn_in_seconds=20.0,
    seed=123,
):
    """
    Simulate multi-sensor acceleration data from a few lightly damped modes.

    The modal equation for each mode is approximately:

        q_ddot + 2*zeta*w*q_dot + w^2*q = u

    Acceleration at sensors is obtained by mixing modal accelerations
    through mode shapes.

    Returns
    -------
    Y : ndarray, shape (n_samples, n_sensors)
        Simulated acceleration response data.
    truth : dict
        True modal frequencies, damping ratios, and mode shapes.
    """

    rng = np.random.default_rng(seed)

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    damping = np.asarray(damping, dtype=float)

    n_modes = len(freqs_hz)
    dt = 1.0 / fs

    # Sensor-space mode shapes
    Phi, sensor_positions = make_bridge_like_mode_shapes(n_sensors, n_modes)

    # Continuous-time state-space model
    #
    # State for each mode:
    #   x_r = [q_r, qdot_r]^T
    #
    # Dynamics:
    #   qdot     = qdot
    #   qddot    = -w^2 q - 2 zeta w qdot + u
    #
    Ac_blocks = []

    for f, zeta in zip(freqs_hz, damping):
        w = 2.0 * np.pi * f

        Ac_r = np.array([
            [0.0, 1.0],
            [-w**2, -2.0 * zeta * w]
        ])

        Ac_blocks.append(Ac_r)

    Ac = splg.block_diag(*Ac_blocks)

    # Input matrix: one independent random force per mode
    Bc = np.zeros((2 * n_modes, n_modes))

    for r in range(n_modes):
        Bc[2 * r + 1, r] = 1.0

    # Output is acceleration at sensors.
    #
    # For one mode:
    #   qddot = -w^2 q - 2 zeta w qdot + u
    #
    # Sensor acceleration:
    #   y = Phi @ qddot
    #
    Cacc = np.zeros((n_sensors, 2 * n_modes))

    for r, (f, zeta) in enumerate(zip(freqs_hz, damping)):
        w = 2.0 * np.pi * f

        Cacc[:, 2 * r] = Phi[:, r] * (-w**2)
        Cacc[:, 2 * r + 1] = Phi[:, r] * (-2.0 * zeta * w)

    Dacc = Phi.copy()

    # Discretize the continuous-time system
    Ad, Bd, Cd, Dd, _ = sps.cont2discrete(
        (Ac, Bc, Cacc, Dacc),
        dt
    )

    n_keep = int(round(duration * fs))
    n_burn = int(round(burn_in_seconds * fs))
    n_total = n_keep + n_burn

    x_state = np.zeros(2 * n_modes)
    Y = np.zeros((n_total, n_sensors))

    # White modal excitation
    U = rng.normal(
        scale=excitation_std,
        size=(n_total, n_modes)
    )

    for k in range(n_total):
        Y[k] = Cd @ x_state + Dd @ U[k]
        x_state = Ad @ x_state + Bd @ U[k]

    # Add sensor noise relative to each channel's standard deviation
    channel_std = np.std(Y, axis=0, keepdims=True)

    Y += rng.normal(
        scale=noise_std * channel_std,
        size=Y.shape
    )

    # Remove burn-in transient
    Y = Y[n_burn:]

    # Remove mean
    Y -= Y.mean(axis=0, keepdims=True)

    truth = {
        "freqs_hz": freqs_hz,
        "damping": damping,
        "mode_shapes": Phi,
        "sensor_positions": sensor_positions,
        "state_matrix": Ad,
    }

    return Y, truth