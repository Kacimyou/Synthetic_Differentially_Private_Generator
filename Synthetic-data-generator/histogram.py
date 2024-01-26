import numpy as np

def modified_histogram_estimator(X, h=0.1, adaptative = True):
    """
    Computes the normalized histogram estimator for multidimensional data with a specified number of bins per axis.

    Parameters:
    - X: numpy array, shape (n, d), where n is the number of data points and d is the dimensionality.
    - h: float, binwidth, where 0 < h < 1. If None, adaptative binwidth will be used.
    - adaptive: bool, whether to use adaptive binwidth or not.

    Returns:
    - fbm: numpy array, the normalized histogram estimator values.

    """
    
    assert h<1 and h>0 , "Error: h must be between 0 and 1"
    
    
    # Calculate the dimensions of the data
    n, d = X.shape

    if adaptative == True:
        # Calculate the binwidth based on the given parameters
        h = n**(-1 / (2 + d))
    
    
    
    # Check if 1/h is an integer, and convert if necessary
    m_inverse = 1 / h
    if not m_inverse.is_integer():
        m_per_axis = int(np.round(m_inverse))
        print(f"Warning: 1/h is not an integer. Converting to the closest integer: {m_per_axis}")
    else:
        m_per_axis = int(m_inverse)
    
    # Initialize the histogram estimator
    fbm_shape = (m_per_axis,) * d
    fbm = np.zeros(fbm_shape)

    # Iterate through each row as a separate data point
    for i in range(n):
        # Determine the bin index for the current data point in each dimension
        bin_indices = tuple(int(np.floor(X[i, j] / h)) for j in range(d))

        # Update the histogram estimator
        fbm[bin_indices] += 1

    # Normalize the histogram estimator
    fbm /= n

    # Check if the sum of elements is equal to 1
    assert np.isclose(np.sum(fbm), 1.0), "Error: Sum of histogram elements is not equal to 1."
    
    return fbm

# Example usage:
# Assuming X is a 3D array with data points
X = np.random.rand(1000, 2)  # Example data with d = 2

fbm_estimator = modified_histogram_estimator(X, adaptative= True)
print(fbm_estimator)


