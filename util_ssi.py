import numpy as np
import scipy.linalg as splg

import rsvd as rsvdfile
import linalg as lg


def estimate_output_covariances(Y, max_lag):
    """
    Estimate output covariance matrices:

        R_lag = E[y_{k+lag} y_k^T]

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_outputs)
        Output-only measurement data.
    max_lag : int
        Maximum covariance lag.

    Returns
    -------
    R : list of ndarray
        R[lag] has shape (n_outputs, n_outputs).
    """

    Y = np.asarray(Y, dtype=float)
    Y = Y - Y.mean(axis=0, keepdims=True)

    n_samples, n_outputs = Y.shape

    R = []

    for lag in range(max_lag + 1):
        if lag == 0:
            R_lag = (Y.T @ Y) / n_samples
        else:
            R_lag = (Y[lag:].T @ Y[:-lag]) / (n_samples - lag)

        R.append(R_lag)
    return R


def estimate_observability_from_hankel(H0, n):
    """
    Estimate extended observability matrix from block Hankel matrix.

    Parameters
    ----------
    H0 : ndarray, shape (i*m, j*m)
        Covariance block Hankel matrix.
    n : int
        Chosen SSI model order.

    Returns
    -------
    O_i : ndarray, shape (i*m, n)
        Estimated extended observability matrix.
    U_n : ndarray
        Truncated left singular vectors.
    s_n : ndarray
        Truncated singular values.
    Vh_n : ndarray
        Truncated right singular vectors, transposed.
    """

    # applying the SVD to the block covariance Hankel matrix
    U, s, Vh = rsvdfile.randomized_hankel_svd(H0, n, oversampling_p=n+10)

    U_n = U[:, :n]
    s_n = s[:n]
    Vh_n = Vh[:n, :]

    # Extended observability matrix estimate
    O_i = U_n * np.sqrt(s_n)[None, :]

    return O_i, U_n, s_n, Vh_n


def build_covariance_hankel(R, block_rows, block_cols, start_lag=1):
    """
    Build covariance block Hankel matrix.

    H0 =
        [ R_1   R_2   R_3   ...
          R_2   R_3   R_4   ...
          R_3   R_4   R_5   ...
          ...   ...   ...   ... ]

    Parameters
    ----------
    R : list of ndarray
        Covariance matrices.
    block_rows : int
        Number of block rows, commonly called i.
    block_cols : int
        Number of block columns, commonly called j.
    start_lag : int
        Usually 1 for SSI-COV.

    Returns
    -------
    H : ndarray
        Block Hankel matrix of shape
        (block_rows * n_outputs, block_cols * n_outputs).
    """

    n_outputs = R[0].shape[0]

    H = np.zeros((
        block_rows * n_outputs,
        block_cols * n_outputs
    ))

    for row in range(block_rows):
        for col in range(block_cols):
            lag = start_lag + row + col

            H[
                row * n_outputs : (row + 1) * n_outputs,
                col * n_outputs : (col + 1) * n_outputs
            ] = R[lag]

    return H


def estimate_A_from_observability(O_i, m):
    """
    Estimate A from the shift property of the observability matrix.
    """

    O_upper = O_i[:-m, :]
    O_lower = O_i[m:, :]

    # get the estimated state matrix A
    A_hat = np.linalg.pinv(O_upper) @ O_lower

    # we also could use the least squares solver by numpy
    # no idea what is better
    """
    A_hat, residuals, rank, singular_values = np.linalg.lstsq(
        O_upper,
        O_lower,
        rcond=None
    )
    """

    return A_hat


def naive_svd(R: list[np.ndarray], block_rows: int, block_cols: int, model_order: int):
    H0 = build_covariance_hankel(
        R,
        block_rows=block_rows,
        block_cols=block_cols,
        start_lag=1
    )

    # SVD of the covariance block Hankel matrix
    U, s, Vh = splg.svd(H0, full_matrices=False)

    n = model_order

    U_n = U[:, :n]
    s_n = s[:n]
    return H0, U_n, s_n, s


def naive_rsvd(R: list[np.ndarray], block_rows: int, block_cols: int, model_order: int):
    H0 = build_covariance_hankel(
        R,
        block_rows=block_rows,
        block_cols=block_cols,
        start_lag=1
    )
    print("Block Hankel Matrix Shape", H0.shape)

    # SVD of the covariance block Hankel matrix
    U, s, Vh = rsvdfile.randomized_hankel_svd(H0, model_order, oversampling_p=10)

    n = model_order

    U_n = U[:, :n]
    s_n = s[:n]
    return H0, U_n, s_n, s



def fast_fft_rsvd(R: list[np.ndarray], block_rows: int, block_cols: int, model_order: int, fast_hankel: bool = True):
    H0 = lg.BlockHankelRepresentation(np.stack(R, axis=0), end_index=block_rows+block_cols+1, window_length=block_rows, window_number=block_cols, fast_hankel=fast_hankel)

    # SVD of the covariance block Hankel matrix
    U, s, Vh = rsvdfile.randomized_hankel_svd(H0, model_order, subspace_iteration_q=0, oversampling_p=10)

    n = model_order

    U_n = U[:, :n]
    s_n = s[:n]
    return H0, U_n, s_n, s


def cov_ssi(Y, fs, block_rows, block_cols, model_order, use_rsvd=True, fast_hankel=True):
    """
    Basic covariance-driven stochastic subspace identification.

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_outputs)
        Measured output data.
    fs : float
        Sampling frequency in Hz.
    block_rows : int
        Number of block rows in the Hankel matrix.
    block_cols : int
        Number of block columns in the Hankel matrix.
    model_order : int
        Chosen state-space model order n.
    fast_hankel : bool
        Whether to use the naive rsvd
    fast_hankel : bool
        Whether to use the faster rsvd (with FFT)

    Returns
    -------
    result : dict
        Identified A, C, observability matrix, poles, frequencies,
        damping ratios, mode shapes, and singular values.
    """

    dt = 1.0 / fs
    n_outputs = Y.shape[1]

    # Need enough covariance lags to fill the Hankel matrix
    max_lag = block_rows + block_cols
    max_lag = max(max_lag, 20)

    # estimate the sequence of covariance matrices
    R = estimate_output_covariances(Y, max_lag=max_lag)

    # decide which svd version to use
    if not use_rsvd:
        H0, U_n, s_n, s = naive_svd(R, block_rows, block_cols, model_order)
    elif use_rsvd and not fast_hankel:
        H0, U_n, s_n, s = naive_rsvd(R, block_rows, block_cols, model_order)
    elif use_rsvd and fast_hankel:
        H0, U_n, s_n, s = fast_fft_rsvd(R, block_rows, block_cols, model_order, fast_hankel=True)
    else:
        raise NotImplementedError

    # Extended observability matrix estimate:
    #
    #   O_i = U_n * Sigma_n^(1/2)
    #
    # Since s_n is a 1D array, multiply each column of U_n
    # by sqrt of the corresponding singular value.
    O_i = U_n * np.sqrt(s_n)[None, :]

    # C is the first output block row of the observability matrix
    C_hat = O_i[:n_outputs, :]

    # Use shift property:
    #
    #   [ C      ]        [ CA     ]
    #   [ CA     ]   A =  [ CA^2   ]
    #   [ CA^2   ]        [ CA^3   ]
    #   [ ...    ]        [ ...    ]
    #
    O_upper = O_i[:-n_outputs, :]
    O_lower = O_i[n_outputs:, :]

    # Solve O_upper @ A ≈ O_lower
    A_hat, residuals, rank, lstsq_svals = np.linalg.lstsq(
        O_upper,
        O_lower,
        rcond=None
    )

    # Eigenvalues of A are the discrete-time poles
    discrete_poles, eigvecs = np.linalg.eig(A_hat)

    # Convert to continuous-time poles
    continuous_poles = np.log(discrete_poles.astype(complex)) / dt

    # Modal parameters
    frequencies_hz = np.abs(np.imag(continuous_poles)) / (2.0 * np.pi)
    damping_ratios = -np.real(continuous_poles) / np.abs(continuous_poles)

    # Mode shapes at measured sensor locations
    mode_shapes = C_hat @ eigvecs

    return {
        "A": A_hat,
        "C": C_hat,
        "O_i": O_i,
        "singular_values": s,
        "discrete_poles": discrete_poles,
        "continuous_poles": continuous_poles,
        "frequencies_hz": frequencies_hz,
        "damping_ratios": damping_ratios,
        "mode_shapes": mode_shapes,
        "H0": H0,
    }