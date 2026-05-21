import time

import numpy as np

import simulate_system as simsys
import util_ssi as ussi


def simulate_data():
    """
    Simulates an acceleration dataset with a number of sensors and excitations.

    Returns
    -------
    Y : ndarray, shape (n_samples, n_sensors)
        Simulated acceleration response data.
    state_matrix : np.ndarray, shape (n_samples, n_sensors)
        True state matrix A for comparison
    fs: int
        The sampling frequency of the simulated data.
    """
    # set a sampling rate
    fs = 200.0

    # simulate the data
    Y, truth = simsys.simulate_acceleration_dataset(
        fs=fs,
        duration=300.0, # 5 min
        freqs_hz=(2.0, 5.0, 8.0),
        damping=(0.01, 0.015, 0.02),
        n_sensors=10,
        excitation_std=1.0,
        noise_std=0.02,
        seed=123,
    )

    # extract the real state matrix
    state_matrix = truth["state_matrix"]
    return Y, state_matrix, fs


def run_ssi(input_data: np.ndarray, fs: int, lag_number: int = 100, model_order: int = 6, use_rsvd: bool = True,
            fast_hankel: bool = True):

    # compute the cov-SSI
    result = ussi.cov_ssi(
        input_data,
        fs=fs,
        block_rows=lag_number,
        block_cols=lag_number,
        model_order=model_order,
        use_rsvd=use_rsvd,
        fast_hankel=fast_hankel,
    )

    # get the state matrix from the result
    state_matrix = result["A"]
    return state_matrix


def run_speed_comparison():

    # get the simulated data
    Y, _, fs = simulate_data()

    # run the cov-SSI for different parametrization
    for lag_number in range(100, 1000, 100):
        for model_order in range(6, min(2*lag_number, 40), 2):
            start = time.perf_counter()
            naive_state_matrix = run_ssi(Y, fs, lag_number, model_order, use_rsvd=False, fast_hankel=False)
            end = time.perf_counter()
            naive_time = end-start
            for fast_hankel in [False, True]:
                start = time.perf_counter()
                estimated_state_matrix = run_ssi(Y, fs, lag_number=lag_number, model_order=model_order, use_rsvd=True, fast_hankel=fast_hankel)
                end = time.perf_counter()
                approximation_error = np.linalg.norm(naive_state_matrix - estimated_state_matrix)
                print(f"cov-SSI with {lag_number=}, {model_order=}, {fast_hankel=} took {end - start:0.4f} seconds (factor={naive_time/(end-start):0.2f}, {approximation_error=})")
            print()


if __name__ == "__main__":
    run_speed_comparison()

