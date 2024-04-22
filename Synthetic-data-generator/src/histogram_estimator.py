import numpy as np
import matplotlib.pyplot as plt
from utils import scale, rescale


def histogram_estimator(X, h=0.1, adaptative=True, method=None, verbose=0):
    """
    Computes the normalized histogram estimator for multidimensional data with a specified number of bins per axis.

    Parameters:
    - X: numpy array, shape (n, d), where n is the number of data points and d is the dimensionality.
    - h: float, binwidth, where 0 < h < 1. If None, adaptative binwidth will be used.
    - adaptive: bool, whether to use adaptive binwidth or not.
    - method: string or None, determine the binwidth of the histogram.

    Returns:
    - hist: numpy array, the normalized histogram estimator values.
    - rescaling_factors: numpy array, rescaling factors to retrieve scale of the initial data

    """

    # Calculate the dimensions of the data
    n, d = X.shape

    X_scaled, rescaling_factors = scale(X)

    # TODO No need to scale when already in 0,1
    """if np.min(X, axis=0) < 0 or np.max(X, axis=0) > 1:
        # Scale the data to [0, 1] range
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    #elif np.min(X, axis=0) >= 0 and np.max(X, axis=0) <= 1:
        
        X_scaled = X
    """

    if adaptative == True or method == None:
        # Calculate the binwidth based on the given parameters
        h = n ** (-1 / (2 + d))

    elif method == "smooth_L2":
        h = n ** (-1 / (2 * d + 3))

    elif method == "smooth_KS":
        h = n ** (-1 / (d + 6))

    elif method == "perturbated":
        h = n ** (-1 / (2 + d))

    assert 0 < h < 1, "Error: h must be between 0 and 1"

    # Check if 1/h is an integer, and convert if necessary
    h_inverse = 1 / h

    if not h_inverse.is_integer():
        m_per_axis = int(np.ceil(h_inverse))
        if verbose == 1:
            print(
                f"Warning: 1/h is not an integer. Converting to the closest integer: {m_per_axis}"
            )
    else:
        m_per_axis = int(h_inverse)

    # Initialize the histogram estimator
    hist_shape = (m_per_axis,) * d
    hist = np.zeros(hist_shape)

    # Iterate through each row as a separate data point
    for i in range(n):
        # Determine the bin index for the current data point in each dimension
        # TODO Fix this
        bin_indices = tuple(
            (
                int(np.floor(X_scaled[i, j] / h)) - 1
                if np.floor(X_scaled[i, j]) == 1
                else int(np.floor(X_scaled[i, j] / h))
            )
            for j in range(d)
        )

        # Update the histogram estimator

        hist[bin_indices] += 1

    # Normalize the histogram estimator
    hist /= n

    # Check if the sum of elements is equal to 1
    assert np.isclose(
        np.sum(hist), 1.0
    ), "Error: Sum of histogram elements is not equal to 1."

    return hist, rescaling_factors


def generate_data_from_hist(
    hist_estimator, k, rescaling_factor=[0, 1], shuffle=True, rescaling=False
):
    """
    Generates synthetic data points according to the empirical distribution represented by fbm_estimator.

    Parameters:
    - hist_estimator: numpy array, the normalized histogram estimator representing the empirical distribution.
    - k: int, the number of data points to generate.

    Returns:
    - synthetic_data: numpy array, the generated synthetic data points.
    """

    # Flatten the hist_estimator to get a 1D array
    flattened_hist = hist_estimator.flatten()

    # Generate m indices according to the empirical distribution
    indices = np.random.choice(len(flattened_hist), size=k, p=flattened_hist)

    # Convert the flat indices to multi-dimensional indices
    multi_dim_indices = np.unravel_index(indices, hist_estimator.shape)

    multi_dim_indices = np.array(multi_dim_indices).T

    # Get the binwidth from the hist_estimator
    binwidth = (
        1 / hist_estimator.shape[0]
    )  # Assuming the binwidth is the same in each dimension

    # Create synthetic data points based on the multi-dimensional indices
    if not (shuffle):
        # TODO : Make a function for rescaling_factor[0] + ((idx+0.5) * binwidth * (rescaling_factor[1] - rescaling_factor[0]))
        synthetic_data = np.array([(idx + 0.5) * binwidth for idx in multi_dim_indices])
    if shuffle:

        synthetic_data = np.array(
            [
                (idx + np.random.uniform(0, 1, size=len(hist_estimator.shape)))
                * binwidth
                for idx in multi_dim_indices
            ]
        )
    if rescaling:

        synthetic_data = rescale(synthetic_data, rescaling_factor)

    return synthetic_data
