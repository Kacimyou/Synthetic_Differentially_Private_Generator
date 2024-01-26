import numpy as np


import numpy as np

def modified_histogram_estimator(X, m_per_axis):
    """
    Computes the normalized histogram estimator for multidimensional data with a specified number of bins per axis.

    Parameters:
    - X: numpy array, shape (n, d), where n is the number of data points and d is the dimensionality.
    - m_per_axis: int, the number of bins per axis.

    Returns:
    - fbm: numpy array, the normalized histogram estimator values.
    - h: float, the binwidth.
    """

    # Calculate the dimensions of the data
    n, d = X.shape

    # Calculate the binwidth based on the given number of bins per axis
    h = 1 / m_per_axis

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

    return fbm, h

"""
# Example usage:
# Assuming X is a 3D array with data points
X = np.random.rand(100, 3)  # Example data with d = 3
m_per_axis = 10  # Example number of bins per axis
fbm_estimator, binwidth = modified_histogram_estimator(X, m_per_axis)

print(f"Number of bins per axis (m_per_axis): {m_per_axis}")
print(f"Binwidth (h): {binwidth}")"""

def modified_histogram_estimator(X):
    """
    Computes the normalized histogram estimator for multidimensional data with a specified number of bins per axis.

    Parameters:
    - X: numpy array, shape (n, d), where n is the number of data points and d is the dimensionality.

    Returns:
    - fbm: numpy array, the normalized histogram estimator values.
    - h: float, the binwidth.
    """



    # Calculate the dimensions of the data
    n, d = X.shape

    
    # Calculate the binwidth based on the given parameters
    h = n**(-1 / (2 + d))
    
    
    # Calculate the number of bins per dimension
    m_per_axis = int(1 / h)
    
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

    return fbm, h

# Example usage:
# Assuming X is a 3D array with data points
X = np.random.rand(100, 3)  # Example data with d = 3

fbm_estimator, binwidth = modified_histogram_estimator(X)

print(f"Number of bins per axis (m_per_axis): {int(1 / binwidth)}")
print(f"Binwidth (h): {binwidth}")



