import numpy as np
import scipy.linalg as splg


def randomized_hankel_svd(hankel_matrix: np.ndarray, k: int, subspace_iteration_q: int = 2, oversampling_p: int = 2):
    """
    Function for the randomized singular vector decomposition using [1].
    Implementation modified from: https://pypi.org/project/fbpca/
    """

    # get the parameter l from the paper
    sample_length_l = k + oversampling_p
    assert 1.25 * sample_length_l < min(hankel_matrix.shape)

    # Apply A to a random matrix, obtaining Q.
    random_matrix_omega = np.random.uniform(low=-1, high=1, size=(hankel_matrix.shape[1], sample_length_l))
    projection_matrix_q = hankel_matrix @ random_matrix_omega

    # Form a matrix Q whose columns constitute a well-conditioned basis for the columns of the earlier Q.
    if subspace_iteration_q == 0:
        (projection_matrix_q, _) = splg.qr(projection_matrix_q, mode='economic')
    if subspace_iteration_q > 0:
        (projection_matrix_q, _) = splg.lu(projection_matrix_q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(subspace_iteration_q):

        # QA
        projection_matrix_q = (projection_matrix_q.T @ hankel_matrix).T

        (projection_matrix_q, _) = splg.lu(projection_matrix_q, permute_l=True)

        # AAQ
        projection_matrix_q = hankel_matrix @ projection_matrix_q

        if it + 1 < subspace_iteration_q:
            (projection_matrix_q, _) = splg.lu(projection_matrix_q, permute_l=True)
        else:
            (projection_matrix_q, _) = splg.qr(projection_matrix_q, mode='economic')

    # SVD Q'*A to obtain approximations to the singular values and right singular vectors of A; adjust the left singular
    # vectors of Q'*A to approximate the left singular vectors of A.
    lower_space_hankel = projection_matrix_q.T @ hankel_matrix
    (R, s, Va) = splg.svd(lower_space_hankel, full_matrices=False)
    U = projection_matrix_q.dot(R)

    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    return U[:, :k], s[:k], Va[:k, :]