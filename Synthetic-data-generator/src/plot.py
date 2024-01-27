#%%
import numpy as np
import matplotlib.pyplot as plt 
from histogram_estimator import generate_data_from_hist, histogram_estimator
from histogram_DP_mechanism import smooth_histogram, perturbed_histogram

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from ipywidgets import interact

#%%

def get_histograms(args_dict):
    # Extract parameters from args_dict
    d = args_dict.get('d', 1)
    n = args_dict.get('n', 10000)
    adaptative = args_dict.get('adaptative', True)
    m = args_dict.get('m', 10000)
    delta = args_dict.get('delta', 0.2)


  
    X = np.random.normal(loc=0, scale=1, size=(n, d))
    #X = np.random.rand(10000, 2)
    # TODO : FIX data type problem with random.rand vs random.normal (chiant)

    # Perform histogram estimation
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Generate synthetic data
    synthetic_data = generate_data_from_hist(hist_estimator, m, rescaling_factors, shuffle= True)

    # Generate DP synthetic data by smoothing
    DP_hist_smoothed = smooth_histogram(hist_estimator, delta = 0.2)
    
    # Generate DP synthetic data by smoothing
    DP_hist_perturbed = perturbed_histogram(hist_estimator, alpha = 0.2, n = n)
    
    smoothed_synthetic_data = generate_data_from_hist(DP_hist_smoothed, m, rescaling_factors, shuffle = True)
    perturbed_synthetic_data = generate_data_from_hist(DP_hist_perturbed, m, rescaling_factors, shuffle = True)
    
    return X, synthetic_data, smoothed_synthetic_data, perturbed_synthetic_data, hist_estimator, rescaling_factors

args_dict = {'adaptative': True, 'm': 5000, 'd': 1, 'n': 1000, 'delta': 0.1}

def histogram_comparaison(args_dict):
    
    X, synthetic_data, smoothed_synthetic_data, perturbed_synthetic_data, hist_estimator, rescaling_factors = get_histograms(args_dict)
    
    bin_number = int(hist_estimator.shape[0])
            
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


# Example usage of the main function

histogram_comparaison(args_dict)


#%%
### CHECK that non DP synthetic data retrieve initial data distribution ##### 

X, synthetic_data, smoothed_synthetic_data, perturbed_synthetic_data, hist_estimator, rescaling_factors = get_histograms({'adaptative': True, 'm': 5000, 'd': 1, 'n': 5000, 'delta': 0.1})


# Sort the data
synthetic_data_sorted = np.sort(synthetic_data[:, 0])
standard_normal_samples_sorted = np.sort(X)

# Create QQ plot
plt.figure(figsize=(8, 8))
plt.scatter(synthetic_data_sorted, standard_normal_samples_sorted,  color='blue')
plt.plot([np.min(X), np.max(X)],
         [np.min(X), np.max(X)],
         color='red', linestyle='--')
plt.title('QQ Plot of Synthetic Data against Standard Normal Distribution')
plt.xlabel('Quantiles of Synthetic Data')
plt.ylabel('Quantiles of Standard Normal Distribution')
plt.show()

#%%




# Define the smoothing_animation function
def smoothing_animation(args_dict):
    # Extract parameters from args_dict
    d = args_dict.get('d')
    n = args_dict.get('n')
    adaptative = args_dict.get('adaptative', True)
    m = args_dict.get('m', 10000)
    delta_values = args_dict.get('delta_values', [0.1])

    # Generate synthetic data for the initial histogram
    X = np.random.normal(loc=0, scale=1, size=(n, d))
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_number = int(hist_estimator.shape[0])

    def update(frame):
        # Clear the previous plot
        ax.clear()

        # Generate synthetic data for the current delta value
        delta = delta_values[frame]
        DP_hist = smooth_histogram(hist_estimator, delta=delta)
        private_synthetic_data = generate_data_from_hist(DP_hist, m, rescaling_factors, shuffle=True)

        # Plot histograms
        ax.hist(X[:, 0], bins=bin_number, alpha=0.5, label='True Data', color='blue', density=True)
        ax.hist(private_synthetic_data[:, 0], bins=bin_number, alpha=0.5, label=f'Synthetic Data (delta={delta})', color='red', density=True)

        ax.legend()
        ax.set_title('Comparison of True and Synthetic Data Histograms')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Density')

    # Create an animation
    animation = FuncAnimation(fig, update, frames=len(delta_values), interval=50, repeat=False)

    # Display the animation
    return HTML(animation.to_jshtml())

# Example usage
args_dict = {'adaptative': True, 'm': 5000, 'd': 1, 'n': 10000, 'delta_values': np.linspace(0,1,100)}
smoothing_animation(args_dict)
   