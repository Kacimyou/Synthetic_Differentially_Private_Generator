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

    print(hist_estimator.shape)
    m = hist_estimator.shape[0]
    d = len(hist_estimator.shape)
    print(m, d)
    
    
    private_hist_estimator = (1 - delta) * hist_estimator + delta/m**d

    # Check if the sum of elements is equal to 1

    print(private_hist_estimator)
    assert np.isclose( np.sum(private_hist_estimator), 1.0), "Error: Sum of private histogram elements is not equal to 1."
    
    return private_hist_estimator


X = np.random.normal(loc=0, scale=1, size=(1000, 2))  # Example data with d = 2

# Perform histogram estimation
hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=True)

# Generate synthetic data
synthetic_data = generate_data_from_hist(hist_estimator, 1000, rescaling_factors, shuffle= True)

DP_hist = smooth_histogram(hist_estimator, delta = 0.2)