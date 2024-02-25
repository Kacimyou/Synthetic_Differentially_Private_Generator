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


#%%
import numpy as np
import matplotlib.pyplot as plt


def haar_wavelet(n, k, t):
    """
    Generate Haar wavelet function for given parameters n and k.
    
    Parameters:
        n (int): Scale parameter.
        k (int): Shift parameter.
        t (float or array-like): Time points at which to evaluate the wavelet function.
    
    Returns:
        array-like: Haar wavelet function evaluated at time points t.
    """
    t = np.asarray(t)
    result = np.zeros_like(t)
    for i, ti in enumerate(t):
        ti_scaled = ti / (2 ** n) - k
        if 0 <= ti_scaled < 0.5:
            result[i] = 1
        elif 0.5 <= ti_scaled < 1:
            result[i] = -1
    return result

# Example usage:
n = 2
k = 1
t = np.linspace(0, 10, 1000)  # Time points
wavelet = haar_wavelet(n, k, t)


# Plot the Haar wavelet
plt.figure(figsize=(10, 6))
plt.plot(t, wavelet, label=f'Haar Wavelet (n={n}, k={k})')
plt.title('Haar Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt



def phi_bar_derivative(j, t):
    """
    Compute the derivative of the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.
    
    Returns:
        array-like: Values of the derivative of the j-th Schauder basis function evaluated at time points t.
    """
    t = np.asarray(t)
    qj = int(np.log2(j))
    pj = j - 2**qj
    
    q2j = int(np.log2(2*j))
    p2j = 2*j - 2**q2j
    
    q2j_plus = int(np.log2(2*j + 1))
    p2j_plus = (2*j + 1) - 2**q2j_plus
    
    
    derivative_values = np.zeros_like(t)
    
    
    for idx, ti in enumerate(t):
        if 0 <= ti <= n:
            if j == 1:
                derivative_values[idx] = 1
            else:
                if   p2j / 2**(q2j-1) <= ti <= (p2j + 1)/ 2**(q2j-1):
                    derivative_values[idx] = 2**qj
                elif p2j_plus / 2**(q2j_plus-1) <= ti <= (p2j_plus + 1)/ 2**(q2j_plus-1):
                    derivative_values[idx] = -2**qj
                else :
                    derivative_values[idx] = 0
    return derivative_values

# Plot the derivative of the Schauder basis functions
num_functions = 2
t = np.linspace(0, 3, 1000)

plt.figure(figsize=(10, 6))
for j in range(1, num_functions + 1):
    derivative_function = phi_bar_derivative(j, t)
    plt.plot(t, derivative_function, label=f'd(φ_{j}(t))/dt')

plt.title('Derivative of Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('d(φ_j(t))/dt')
plt.grid(True)
plt.legend()
plt.show()


def is_in_I_j(j, t):
    """
    Check if t is within the interval I_j for a given integer j.
    
    Parameters:
        j (int): Index of the interval.
        t (float): Value to check.
    
    Returns:
        bool: True if t is within I_j, False otherwise.
    """
    
    pj,qj = get_pj_qj(j)

    lower_bound = pj  / (2 ** (qj-1))
    upper_bound = (pj+1) / (2 ** (qj-1))
    print(lower_bound, upper_bound, "qj=", qj,"pj=", pj)
    return lower_bound <= t < upper_bound

def phi_bar(j, t):
    """
    Compute the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.
    
    Returns:
        array-like: Values of the j-th Schauder basis function evaluated at time points t.
    """
    t = np.asarray(t)
    pj,qj = get_pj_qj(j)

    phi_values = np.zeros_like(t)
    
    for idx, ti in enumerate(t):
        if j == 1:
            phi_values[idx] = ti
        else:
            if is_in_I_j(2*j, ti):
                phi_values[idx] = 2**qj * (ti - pj / 2**(qj - 1))
            elif is_in_I_j(2*j + 1, ti):
                phi_values[idx] = 2**qj * ((pj + 1) / 2**(qj - 1) - ti)
            else:
                phi_values[idx] = 0

    return phi_values

# Plot the Schauder basis functions
num_functions = 3
t = np.linspace(0, 3, 1000)

plt.figure(figsize=(10, 6))
for j in range(1, num_functions + 1):
    basis_function_j = phi_bar(j, t)
    plt.plot(t, basis_function_j, label=f'φ_{j}(t)')

plt.title('Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('φ_j(t)')
plt.grid(True)
plt.legend()
plt.show()