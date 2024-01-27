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
    
    dp_hist = (1 - delta) * hist_estimator + delta/bin_number**d

    # Check if the sum of elements is equal to 1
    assert np.isclose( np.sum(dp_hist), 1.0), "Error: Sum of private histogram elements is not equal to 1."
    
    return dp_hist

def perturbed_histogram(hist_estimator, alpha, n):
    
    """
    Add Laplacian noise to each component of the histogram estimator.

    Parameters:
    - hist_estimator: numpy array, the normalized histogram estimator.
    - alpha: float, privacy parameter.
    - n : int, number of samples.

    Returns:
    - dp_hist: numpy array, differentially private histogram estimator.
    """
    #sensitivity = 1 / n  # Sensitivity of the histogram estimator

    # Generate Laplace noise for each component
    laplace_noise = np.random.laplace(scale= 8/ alpha^2 , size=hist_estimator.shape)

    # Add Laplace noise to the histogram estimator
    dp_hist = hist_estimator + laplace_noise / n

    # Clip to ensure non-negativity
    dp_hist = np.clip(dp_hist, 0, np.inf)

    # Normalize the differentially private histogram estimator
    dp_hist /= np.sum(dp_hist)
    
    # Check if the sum of elements is equal to 1
    assert np.isclose( np.sum(dp_hist), 1.0), "Error: Sum of private histogram elements is not equal to 1."
    

    return dp_hist
    
    
