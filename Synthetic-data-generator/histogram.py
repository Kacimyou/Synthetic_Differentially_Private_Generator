#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%
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
        bin_indices = tuple(int(np.floor(X_scaled[i, j] / h)) for j in range(d))
        
        # Update the histogram estimator
        fbm[bin_indices] += 1


    # Normalize the histogram estimator
    fbm /= n
    #TODO CONTINUE the implementation of privacy options

    # Check if the sum of elements is equal to 1
    assert np.isclose(np.sum(fbm), 1.0), "Error: Sum of histogram elements is not equal to 1."
    
    
    rescaling_factors = np.array([np.min(X, axis=0), np.max(X, axis=0)])
    

    return fbm, rescaling_factors



def generate_data_from_fbm(fbm_estimator, m, rescaling_factor = [1,0], shuffle = False):
    """
    Generates synthetic data points according to the empirical distribution represented by fbm_estimator.

    Parameters:
    - fbm_estimator: numpy array, the normalized histogram estimator representing the empirical distribution.
    - m: int, the number of data points to generate.

    Returns:
    - synthetic_data: numpy array, the generated synthetic data points.
    """

    # Flatten the fbm_estimator to get a 1D array
    flattened_fbm = fbm_estimator.flatten()

    # Generate m indices according to the empirical distribution
    indices = np.random.choice(len(flattened_fbm), size=m, p=flattened_fbm)

    # Convert the flat indices to multi-dimensional indices
    multi_dim_indices = np.unravel_index(indices, fbm_estimator.shape)
    


    # Get the binwidth from the fbm_estimator
    binwidth = 1 / fbm_estimator.shape[0]  # Assuming the binwidth is the same in each dimension
    
    # Create synthetic data points based on the multi-dimensional indices
    if not(shuffle):
        synthetic_data = np.column_stack(
        [rescaling_factor[0] + ((idx+0.5) * binwidth * (rescaling_factor[1] - rescaling_factor[0])) for idx in multi_dim_indices]
    )
    if shuffle:
        # Generate a list of random numbers
        random_noise = np.random.uniform(0, 1, size=m)

        #TODO FIX noise problem check with QQ PLOT
        print([(idx + noise) for idx, noise in zip(multi_dim_indices, random_noise)])
        
        
        # Use the list of random numbers within the list comprehension
        synthetic_data = np.column_stack(
            [rescaling_factor[0] + ((idx + noise) * binwidth * (rescaling_factor[1] - rescaling_factor[0])) for idx, noise in zip(multi_dim_indices, random_noise)]
        )
    

    return synthetic_data



def main(args_dict):
    # Extract parameters from args_dict
    d = args_dict.get('d')
    n = args_dict.get('n')
    adaptative = args_dict.get('adaptative', True)
    m = args_dict.get('m', 10000)


  
    X = np.random.normal(loc=0, scale=1, size=(n, d))  # Example data with d = 2
    #X = np.random.rand(10000, 2)
    # TODO : FIX data type problem with random.rand vs random.normal (chiant)

    # Perform histogram estimation
    fbm_estimator, rescaling_factors = modified_histogram_estimator(X, adaptative=adaptative)

    # Generate synthetic data
    synthetic_data = generate_data_from_fbm(fbm_estimator, m, rescaling_factors, shuffle= True)


    bin_number = int( fbm_estimator.shape[0])
    
    
    # Plot histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot true data histogram
    ax.hist(X[:, 0], bins=bin_number, alpha=0.5, label='True Data', color='blue', density=True)
    # Plot synthetic data histogram
    ax.hist(synthetic_data[:, 0], bins=bin_number, alpha=0.5, label='Synthetic Data', color='orange', density=True)
    ax.legend()
    ax.set_title('Comparison of True and Synthetic Data Histograms')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Density')

    plt.show()

    return X, synthetic_data, fbm_estimator, rescaling_factors

# Example usage of the main function
args_dict = {'adaptative': True, 'm': 500, 'd': 1, 'n': 500}
X, synthetic_data, fbm_estimator, rescaling_factors = main(args_dict)



# %%

synthetic_data_2 = generate_data_from_fbm(fbm_estimator, 10000, rescaling_factors, shuffle= True)

plt.hist(synthetic_data_2 , bins= 22)
print(len(np.unique(synthetic_data_2)))


# %%




# Use a random normal distribution for comparison
standard_normal_samples = np.random.normal(size=synthetic_data_2.shape[0])

# Sort the data
synthetic_data_sorted = np.sort(synthetic_data_2[:, 0])
standard_normal_samples_sorted = np.sort(standard_normal_samples)

# Create QQ plot
plt.figure(figsize=(8, 8))
plt.scatter(synthetic_data_sorted, standard_normal_samples_sorted,  color='blue')
plt.plot([np.min(standard_normal_samples), np.max(standard_normal_samples)],
         [np.min(standard_normal_samples), np.max(standard_normal_samples)],
         color='red', linestyle='--')
plt.title('QQ Plot of Synthetic Data against Standard Normal Distribution')
plt.xlabel('Quantiles of Synthetic Data')
plt.ylabel('Quantiles of Standard Normal Distribution')
plt.show()

print(synthetic_data_sorted[:10])

# %%


modified_histogram_estimator