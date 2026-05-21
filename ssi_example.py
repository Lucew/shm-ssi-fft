import numpy as np
import matplotlib.pyplot as plt

import simulate_system as simsys
import util_ssi as ussi


def extract_oscillatory_modes(
    ssi_result,
    fs,
    min_freq=0.1,
    max_freq=None,
    max_damping=0.20,
):
    """
    Extract one pole from each complex-conjugate pole pair.

    We keep only poles with positive imaginary part.
    """

    if max_freq is None:
        max_freq = 0.5 * fs

    poles = ssi_result["continuous_poles"]
    freqs = ssi_result["frequencies_hz"]
    damping = ssi_result["damping_ratios"]
    mode_shapes = ssi_result["mode_shapes"]

    keep = (
        (np.imag(poles) > 0.0)
        & (freqs >= min_freq)
        & (freqs <= max_freq)
        & np.isfinite(damping)
        & (damping > 0.0)
        & (damping < max_damping)
    )

    idx = np.where(keep)[0]

    # Sort by frequency
    idx = idx[np.argsort(freqs[idx])]

    return freqs[idx], damping[idx], mode_shapes[:, idx], idx


def stabilization_scan(
    Y,
    fs,
    block_rows,
    block_cols,
    model_orders,
    min_freq=0.1,
    max_freq=20.0,
    max_damping=0.20,
):
    """
    Run SSI-COV for many model orders and collect identified poles.
    Useful for a simple stabilization-style frequency/order plot.
    """

    all_orders = []
    all_freqs = []
    all_damping = []

    for n in model_orders:
        result = ussi.cov_ssi(
            Y,
            fs=fs,
            block_rows=block_rows,
            block_cols=block_cols,
            model_order=n
        )

        f, zeta, _, _ = extract_oscillatory_modes(
            result,
            fs=fs,
            min_freq=min_freq,
            max_freq=max_freq,
            max_damping=max_damping
        )

        all_orders.extend([n] * len(f))
        all_freqs.extend(f)
        all_damping.extend(zeta)

    return np.array(all_orders), np.array(all_freqs), np.array(all_damping)


def main():
    fs = 100.0

    Y, truth = simsys.simulate_acceleration_dataset(
        fs=fs,
        duration=300.0,
        freqs_hz=(2.0, 5.0, 8.0),
        damping=(0.01, 0.015, 0.02),
        n_sensors=10,
        excitation_std=1.0,
        noise_std=0.02,
        seed=123,
    )

    print("Data shape:", Y.shape)
    print("True frequencies [Hz]:", truth["freqs_hz"])
    print("True damping ratios:", truth["damping"])

    # SSI settings
    block_rows = 100
    block_cols = 100

    # For 3 physical modes, a minimal real-valued state order is about 2 * 3 = 6.
    # In real applications, you normally scan many model orders instead.
    model_order = 6

    result = ussi.cov_ssi(
        Y,
        fs=fs,
        block_rows=block_rows,
        block_cols=block_cols,
        model_order=model_order
    )

    f_id, zeta_id, phi_id, idx = extract_oscillatory_modes(
        result,
        fs=fs,
        min_freq=0.1,
        max_freq=20.0,
        max_damping=0.20
    )

    print("\nIdentified modes from model order", model_order)
    print("-------------------------------------")

    for k, (f, zeta) in enumerate(zip(f_id, zeta_id), start=1):
        print(f"Mode {k}: f = {f:8.4f} Hz, damping = {100 * zeta:6.3f} %")

    # Plot singular values
    plt.figure()
    plt.semilogy(result["singular_values"], marker="o")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title("SVD of covariance block Hankel matrix")
    plt.grid(True)
    plt.show()

    # Plot first sensor signal
    t = np.arange(Y.shape[0]) / fs

    plt.figure()
    plt.plot(t[:3000], Y[:3000, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration, sensor 1")
    plt.title("Simulated acceleration data")
    plt.grid(True)
    plt.show()

    # Simple stabilization-style scan
    model_orders = np.arange(2, 31, 2)

    orders, freqs, damping = stabilization_scan(
        Y,
        fs=fs,
        block_rows=block_rows,
        block_cols=block_cols,
        model_orders=model_orders,
        min_freq=0.1,
        max_freq=15.0,
        max_damping=0.20
    )

    plt.figure()
    plt.scatter(freqs, orders, s=18)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Model order")
    plt.title("Simple SSI-COV stabilization-style diagram")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
