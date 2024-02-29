# %%
import numpy as np
import matplotlib.pyplot as plt
from histogram_estimator import generate_data_from_hist, histogram_estimator
from histogram_DP_mechanism import smooth_histogram, perturbed_histogram
from random_walk import phi_bar, psi_bar_j

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from ipywidgets import interact

# %%


def get_histograms(args_dict):
    # Extract parameters from args_dict
    d = args_dict.get("d", 1)
    n = args_dict.get("n", 10000)
    adaptative = args_dict.get("adaptative", True)
    m = args_dict.get("m", 10000)
    delta = args_dict.get("delta", 0.2)
    alpha = args_dict.get("alpha", 0.2)

    X = np.random.normal(loc=0, scale=1, size=(n, d))
    # X = np.random.rand(10000, 2)
    # Set mean and covariance matrix
    mean = [0, 1]
    covariance_matrix = [[0.1, 0.4], [0.4, 1]]

    # Generate random sample
    #X = np.random.multivariate_normal(mean, covariance_matrix, size=n)
    # TODO : FIX data type problem with random.rand vs random.normal (chiant)

    # Perform histogram estimation
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Generate synthetic data
    synthetic_data = generate_data_from_hist(
        hist_estimator, m, rescaling_factors, shuffle=True
    )

    # Generate DP synthetic data by smoothing
    DP_hist_smoothed = smooth_histogram(hist_estimator, delta=delta)

    # Generate DP synthetic data by smoothing
    DP_hist_perturbed = perturbed_histogram(hist_estimator, alpha=alpha, n=n)

    smoothed_synthetic_data = generate_data_from_hist(
        DP_hist_smoothed, m, rescaling_factors, shuffle=True
    )
    perturbed_synthetic_data = generate_data_from_hist(
        DP_hist_perturbed, m, rescaling_factors, shuffle=True
    )

    return (
        X,
        synthetic_data,
        smoothed_synthetic_data,
        perturbed_synthetic_data,
        hist_estimator,
        rescaling_factors,
    )


def histogram_comparaison(args_dict, privacy):

    assert privacy in [
        None,
        "smooth",
        "perturbed",
    ], "privacy parameter should be in equal to None, smooth or perturbed"

    (
        X,
        synthetic_data,
        smoothed_synthetic_data,
        perturbed_synthetic_data,
        hist_estimator,
        rescaling_factors,
    ) = get_histograms(args_dict)

    bin_number = int(hist_estimator.shape[0])

    # Plot histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot true data histogram
    ax.hist(
        X[:, 0],
        bins=bin_number,
        alpha=0.5,
        label="True Data",
        color="blue",
        density=True,
    )
    if privacy == None:
        # Plot synthetic data histogram
        ax.hist(
            synthetic_data[:, 0],
            bins=bin_number,
            alpha=0.5,
            label="Synthetic Data",
            color="orange",
            density=True,
        )

    if privacy == "smooth":
        # Plot smoothed synthetic data histogram
        ax.hist(
            smoothed_synthetic_data[:, 0],
            bins=bin_number,
            alpha=0.5,
            label="Smoothed Synthetic Data",
            color="red",
            density=True,
        )

    if privacy == "perturbed":
        # Plot smoothed synthetic data histogram
        ax.hist(
            perturbed_synthetic_data[:, 0],
            bins=bin_number,
            alpha=0.5,
            label="Perturbed Synthetic Data",
            color="red",
            density=True,
        )

    ax.legend()
    ax.set_title("Comparison of True and Synthetic Data Histograms")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Density")
    plt.show()


# Example usage of the main function

args_dict = {
    "adaptative": True,
    "m": 10000,
    "d": 2,
    "n": 10000,
    "delta": 0.1,
    "alpha": 0.5,
}

histogram_comparaison(args_dict, privacy="perturbed")


# %%
### CHECK that non DP synthetic data retrieve initial data distribution #####

(
    X,
    synthetic_data,
    smoothed_synthetic_data,
    perturbed_synthetic_data,
    hist_estimator,
    rescaling_factors,
) = get_histograms({"adaptative": True, "m": 1000, "d": 1, "n": 1000, "delta": 0.1})

# Sort the data
synthetic_data_sorted = np.sort(synthetic_data[:, 0])
standard_normal_samples_sorted = np.sort(
    np.random.normal(size=len(synthetic_data_sorted))
)
X_sorted = np.sort(X[:, 0])


# Create QQ plot
plt.figure(figsize=(8, 8))
plt.scatter(synthetic_data_sorted, X_sorted, color="blue")
plt.plot(
    [np.min(X[:, 0]), np.max(X[:, 0])],
    [np.min(X[:, 0]), np.max(X[:, 0])],
    color="red",
    linestyle="--",
)
plt.title("QQ Plot of Synthetic Data against Standard Normal Distribution")
plt.xlabel("Quantiles of Synthetic Data")
plt.ylabel("Quantiles of Standard Normal Distribution")
plt.show()

# %%


# Define the smoothing_animation function
def smoothing_animation(args_dict, privacy):
    assert privacy in [
        "smooth",
        "perturbed",
    ], "privacy parameter should be in equal to smooth or perturbed"
    # Extract parameters from args_dict
    d = args_dict.get("d")
    n = args_dict.get("n")
    adaptative = args_dict.get("adaptative", True)
    m = args_dict.get("m", 10000)
    delta_values = args_dict.get("delta_values", [0.1])
    alpha_values = args_dict.get("alpha_values", [0.1])

    # Generate synthetic data for the initial histogram
    X = np.random.normal(loc=0, scale=1, size=(n, d))
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_number = int(hist_estimator.shape[0])

    def update(frame):
        # Clear the previous plot
        ax.clear()

        if privacy == "smooth":
            # Generate synthetic data for the current delta value
            DP_param = delta_values[frame]
            DP_hist = smooth_histogram(hist_estimator, delta=DP_param)
            private_synthetic_data = generate_data_from_hist(
                DP_hist, m, rescaling_factors, shuffle=True
            )
        if privacy == "perturbed":
            # Generate synthetic data for the current delta value
            DP_param = alpha_values[frame]
            DP_hist = perturbed_histogram(hist_estimator, alpha=DP_param, n=n)
            private_synthetic_data = generate_data_from_hist(
                DP_hist, m, rescaling_factors, shuffle=True
            )

        # Plot histograms
        ax.hist(
            X[:, 0],
            bins=bin_number,
            alpha=0.5,
            label="True Data",
            color="blue",
            density=True,
        )
        ax.hist(
            private_synthetic_data[:, 0],
            bins=bin_number,
            alpha=0.5,
            label=f"Synthetic Data (DP_param={DP_param})",
            color="red",
            density=True,
        )

        ax.legend()
        ax.set_title("Comparison of True and Synthetic Data Histograms")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Density")

    # Create an animation
    if privacy == "smooth":
        animation = FuncAnimation(
            fig, update, frames=len(delta_values), interval=50, repeat=False
        )
    if privacy == "perturbed":
        animation = FuncAnimation(
            fig, update, frames=len(alpha_values), interval=50, repeat=False
        )
    # Display the animation
    return HTML(animation.to_jshtml())


# Example usage
args_dict = {
    "adaptative": True,
    "m": 50000,
    "d": 1,
    "n": 100000,
    "delta_values": np.linspace(0, 1, 100),
    "alpha_values": np.linspace(0.01, 0.2, 100),
}
smoothing_animation(args_dict, privacy="perturbed")


# %%
(
    X,
    synthetic_data,
    smoothed_synthetic_data,
    perturbed_synthetic_data,
    hist_estimator,
    rescaling_factors,
) = get_histograms(
    {"adaptative": True, "m": 2000, "d": 1, "n": 11000, "delta": 0.2, "alpha": 0.35}
)


# Scatter plots with labels
plt.scatter(X[:, 0], X[:, 1], label="Original Data")
plt.scatter(
    smoothed_synthetic_data[:, 0],
    smoothed_synthetic_data[:, 1],
    label="Smoothed Synthetic Data",
)
plt.scatter(
    perturbed_synthetic_data[:, 0],
    perturbed_synthetic_data[:, 1],
    label="Perturbed Synthetic Data",
)

# Add legend
plt.legend()

# Show the plot
plt.show()
print(np.shape(synthetic_data), np.shape(X))

# %%
# Plot the Schauder basis functions
num_functions = 8
t = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for j in range(1, num_functions + 1):
    basis_function_j = phi_bar(j, t)
    plt.plot(t, basis_function_j, label=f"φ_{j}(t)")

plt.title("Schauder Basis Functions")
plt.xlabel("t")
plt.ylabel("φ_j(t)")
plt.grid(True)
plt.legend()
plt.show()

# %%
# Plot the Haar basis functions
num_functions = 9
t = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for j in range(1, num_functions + 1):
    basis_function_j = psi_bar_j(j, t)
    plt.plot(t, basis_function_j, label=f"φ_{j}(t)")

plt.title("Haar Basis Functions")
plt.xlabel("t")
plt.ylabel("φ_j(t)")
plt.grid(True)
plt.legend()
plt.show()
