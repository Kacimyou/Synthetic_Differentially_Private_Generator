#%%
import numpy as np
import matplotlib.pyplot as plt 
from histogram_estimator import generate_data_from_fbm, modified_histogram_estimator


#%%

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
args_dict = {'adaptative': True, 'm': 5000, 'd': 1, 'n': 1000}
X, synthetic_data, fbm_estimator, rescaling_factors = main(args_dict)



synthetic_data_2 = generate_data_from_fbm(fbm_estimator, 10000, rescaling_factors, shuffle= True)

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

