import numpy as np
from histogram_estimator import generate_data_from_hist, histogram_estimator


def smooth_histogram(hist_estimator, delta):
    """
    Smooth a given histogram estimator to achieve differential privacy.

    Parameters:
    - hist_estimator: numpy array, the original histogram estimator.
    - delta: float, privacy parameter.

    Returns:
    - private_hist_estimator: numpy array, the differentially private histogram estimator.
    """

    # Apply differential privacy mechanism

    bin_number = hist_estimator.shape[0]
    d = len(hist_estimator.shape)

    dp_hist = (1 - delta) * hist_estimator + delta / bin_number**d

    # Check if the sum of elements is equal to 1
    assert np.isclose(
        np.sum(dp_hist), 1.0
    ), "Error: Sum of private histogram elements is not equal to 1."

    return dp_hist


def perturbed_histogram(hist_estimator, epsilon, n):
    """
    Add Laplacian noise to each component of the histogram estimator.

    Parameters:
    - hist_estimator: numpy array, the normalized histogram estimator.
    - epsilon: float, privacy parameter.
    - n : int, number of samples.

    Returns:
    - dp_hist: numpy array, differentially private histogram estimator.
    """
    assert 1 > epsilon > 0, "Error: epsilon should be between 0 and 1"
    # sensitivity = 1 / n  # Sensitivity of the histogram estimator

    # Generate Laplace noise for each component

    laplace_noise = np.random.laplace(scale=8 / epsilon**2, size=hist_estimator.shape)

    # Add Laplace noise to the histogram estimator
    dp_hist = hist_estimator + laplace_noise / n

    # Clip to ensure non-negativity
    dp_hist = np.clip(dp_hist, 0, np.inf)

    # Normalize the differentially private histogram estimator
    dp_hist /= np.sum(dp_hist)

    # Check if the sum of elements is equal to 1
    assert np.isclose(
        np.sum(dp_hist), 1.0
    ), "Error: Sum of private histogram elements is not equal to 1."

    return dp_hist


def generate_smooth_data(X, k, delta, adaptative, shuffle=True):

    # Perform histogram estimation
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Generate DP synthetic data by smoothing
    DP_hist_smoothed = smooth_histogram(hist_estimator, delta=delta)

    smoothed_synthetic_data = generate_data_from_hist(
        DP_hist_smoothed, k=k, rescaling_factor=rescaling_factors, shuffle=shuffle
    )

    return smoothed_synthetic_data


def generate_perturbated_data(X, k, epsilon, adaptative=True, shuffle=True):

    n, d = X.shape

    # Perform histogram estimation
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Generate DP synthetic data by perturbation
    DP_hist_perturbed = perturbed_histogram(hist_estimator, epsilon=epsilon, n=n)

    perturbed_synthetic_data = generate_data_from_hist(
        DP_hist_perturbed, k=k, rescaling_factor=rescaling_factors, shuffle=shuffle
    )

    return perturbed_synthetic_data, hist_estimator
