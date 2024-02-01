#%%
import numpy as np 
import matplotlib.pyplot as plt
#%%

prob = np.array([0.006, 0.05,  0.198, 0.36,  0.272, 0.108, 0.004, 0.002])


print(np.sum(prob))

delta = 0.1
m = len(prob)


# Set mean and covariance matrix
mean = [0, 1]
covariance_matrix = [[0.1, 0.4], [0.4, 1]]

# Generate random sample
random_sample = np.random.multivariate_normal(mean, covariance_matrix, size=1000)





plt.scatter(random_sample[:,0],random_sample[:,1])

# %%


import numpy as np

# Set mean and covariance matrix
mean = [0, 1]
covariance_matrix = [[10, 0.2], [0.2, 1]]

# Generate random sample
random_sample = np.random.multivariate_normal(mean, covariance_matrix, size=1000)

# Calculate min and max along each axis
min_values = np.min(random_sample, axis=0)
max_values = np.max(random_sample, axis=0)

# Store min and max values in rescaling_factors array
rescaling_factors = np.vstack((min_values, max_values))

X = (random_sample - rescaling_factors[0]) / (rescaling_factors[1] - rescaling_factors[0])
# Print rescaling_factors array
print(rescaling_factors, np.min(X, axis =0))
# %%

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
    
    # Calculate min and max along each axis
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    rescaling_factors = np.vstack((min_values, max_values))

    """if np.min(X, axis=0) < 0 or np.max(X, axis=0) > 1:
        # Scale the data to [0, 1] range
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    #elif np.min(X, axis=0) >= 0 and np.max(X, axis=0) <= 1:
        
        X_scaled = X
    """
    
    
    X_scaled = (X - rescaling_factors[0]) / (rescaling_factors[1] - rescaling_factors[0])
    
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
    
    test_multi = np.array(multi_dim_indices).T

    # Get the binwidth from the hist_estimator
    binwidth = 1 / hist_estimator.shape[0]  # Assuming the binwidth is the same in each dimension
    print(hist_estimator.shape)
    # Create synthetic data points based on the multi-dimensional indices

    print("rescaling_factor 0 vaut:",rescaling_factor[0])
    # Create synthetic data points based on the multi-dimensional indices
    if not(shuffle):
        #TODO : Make a function for rescaling_factor[0] + ((idx+0.5) * binwidth * (rescaling_factor[1] - rescaling_factor[0]))
        synthetic_data = np.array(
        [rescaling_factor[0] + ((idx+0.5) * binwidth * (rescaling_factor[1] - rescaling_factor[0])) for idx in test_multi]
    )
    if shuffle:
        
        synthetic_data = np.array(
            [rescaling_factor[0] + ((idx + np.random.uniform(0, 1, size = len(hist_estimator.shape))) * binwidth * (rescaling_factor[1] - rescaling_factor[0])) for idx in test_multi]
        )
    
    return synthetic_data



mean = [0, 1]
covariance_matrix = [[10, 0.4], [0.4, 1]]
X = np.random.rand(10000, 4)
# Generate random sample
#X = np.random.multivariate_normal(mean, covariance_matrix, size=100)
hist, rescale = histogram_estimator(X)
print(rescale)

# %%

synthetic_data = generate_data_from_hist(hist, 1000, rescale)
synthetic_data
#%%
np.array([1,1]) *np.array([2,5])