
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def histogram_estimator(X, h=0.1, adaptative = True):
    """
    Computes the normalized histogram estimator for multidimensional data with a specified number of bins per axis.

    Parameters:
    - X: numpy array, shape (n, d), where n is the number of data points and d is the dimensionality.
    - h: float, binwidth, where 0 < h < 1. If None, adaptative binwidth will be used.
    - adaptive: bool, whether to use adaptive binwidth or not.

    Returns:
    - hist: numpy array, the normalized histogram estimator values.

    """
    
    
    # Calculate the dimensions of the data
    n, d = X.shape

    """if np.min(X, axis=0) < 0 or np.max(X, axis=0) > 1:
        # Scale the data to [0, 1] range
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    #elif np.min(X, axis=0) >= 0 and np.max(X, axis=0) <= 1:
        
        X_scaled = X
    """
    
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    if adaptative == True:
        # Calculate the binwidth based on the given parameters
        h = n**(-1 / (2 + d))
    
    assert h<1 and h>0 , "Error: h must be between 0 and 1"
    
    
    # Check if 1/h is an integer, and convert if necessary
    m_inverse = 1 / h
    if not m_inverse.is_integer():
        m_per_axis = int(np.ceil(m_inverse))
        print(f"Warning: 1/h is not an integer. Converting to the closest integer: {m_per_axis}")
    else:
        m_per_axis = int(m_inverse)
        

    

    # Initialize the histogram estimator
    hist_shape = (m_per_axis,) * d
    hist = np.zeros(hist_shape)

    # Iterate through each row as a separate data point
    for i in range(n):
        # Determine the bin index for the current data point in each dimension
        bin_indices = tuple(int(np.floor(X_scaled[i, j] / h)) for j in range(d))
        
        # Update the histogram estimator
        hist[bin_indices] += 1


    # Normalize the histogram estimator
    hist /= n
    #TODO CONTINUE the implementation of privacy options

    # Check if the sum of elements is equal to 1
    assert np.isclose(np.sum(hist), 1.0), "Error: Sum of histogram elements is not equal to 1."
    
    
    rescaling_factors = np.array([np.min(X, axis=0), np.max(X, axis=0)])
    

    return hist, rescaling_factors



def generate_data_from_hist(hist_estimator, m, rescaling_factor = [1,0], shuffle = True):
    """
    Generates synthetic data points according to the empirical distribution represented by fbm_estimator.

    Parameters:
    - hist_estimator: numpy array, the normalized histogram estimator representing the empirical distribution.
    - m: int, the number of data points to generate.

    Returns:
    - synthetic_data: numpy array, the generated synthetic data points.
    """

    # Flatten the hist_estimator to get a 1D array
    flattened_hist = hist_estimator.flatten()

    # Generate m indices according to the empirical distribution
    indices = np.random.choice(len(flattened_hist), size=m, p=flattened_hist)

    # Convert the flat indices to multi-dimensional indices
    multi_dim_indices = np.unravel_index(indices, hist_estimator.shape)
    


    # Get the binwidth from the hist_estimator
    binwidth = 1 / hist_estimator.shape[0]  # Assuming the binwidth is the same in each dimension
    
    # Create synthetic data points based on the multi-dimensional indices
    
    if not(shuffle):
        synthetic_data = np.column_stack(
        [rescaling_factor[0] + ((idx+0.5) * binwidth * (rescaling_factor[1] - rescaling_factor[0])) for idx in multi_dim_indices]
    )
    if shuffle:
        # Generate a list of random numbers
        random_noise = np.random.uniform(0, 1, size=m)

        # Use the list of random numbers within the list comprehension
        synthetic_data = np.column_stack(
            [[rescaling_factor[0] + ((idx + np.random.uniform(0, 1)) * binwidth * (rescaling_factor[1] - rescaling_factor[0])) for idx in row] for row in multi_dim_indices]

        )

    return synthetic_data



