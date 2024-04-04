# %%
import numpy as np
from histogram_estimator import generate_data_from_hist, histogram_estimator
import matplotlib.pyplot as plt


def smooth_histogram(hist_estimator, delta):
    """
    Smooth a given histogram estimator to achieve differential privacy.

    Parameters:
    - hist_estimator: numpy array, the original histogram estimator.
    - delta: float, privacy parameter.

    Returns:
    - private_hist_estimator: numpy array, the differentially private histogram estimator.
    """
    assert 1 > delta > 0, "Error: delta should be between 0 and 1."
    # Apply differential privacy mechanism

    bin_number = hist_estimator.shape[0]
    d = len(hist_estimator.shape)

    dp_hist = (1 - delta) * hist_estimator + delta / bin_number**d

    # Check if the sum of elements is equal to 1
    assert np.isclose(
        np.sum(dp_hist), 1.0
    ), "Error: Sum of private histogram elements is not equal to 1."

    return dp_hist


def perturbed_histogram(hist_estimator, epsilon, n):
    """
    Add Laplacian noise to each component of the histogram estimator.

    Parameters:
    - hist_estimator: numpy array, the normalized histogram estimator.
    - epsilon: float, privacy parameter.
    - n : int, number of samples.

    Returns:
    - dp_hist: numpy array, differentially private histogram estimator.
    """
    assert epsilon > 0, "Error: epsilon should be strictly positive"
    # sensitivity = 1 / n  # Sensitivity of the histogram estimator

    # Generate Laplace noise for each component

    laplace_noise = np.random.laplace(scale=8 / epsilon**2, size=hist_estimator.shape)

    # Add Laplace noise to the histogram estimator
    dp_hist = hist_estimator + laplace_noise / n

    # Clip to ensure non-negativity
    dp_hist = np.clip(dp_hist, 0, np.inf)

    # Normalize the differentially private histogram estimator
    dp_hist /= np.sum(dp_hist)

    # Check if the sum of elements is equal to 1
    assert np.isclose(
        np.sum(dp_hist), 1.0
    ), "Error: Sum of private histogram elements is not equal to 1."

    return dp_hist


def generate_smooth_data(
    X, k, epsilon, adaptative=True, shuffle=True, norm="L2", automatic=True
):

    if norm == "L2":

        m = int(np.ceil(n ** (1 / (2 * d + 3)))) ** d
        print(m)
        if automatic:
            k = int(n ** ((d + 2) / (2 * d + 3)))  # k must be int
            delta = n ** (-1 / (d + 3))
            print(
                "Parameters k, delta and m where chosen to meet privacy:\n",
                "k = ",
                k,
                "\n",
                "delta = ",
                delta,
                "\n",
                "m = ",
                m,
                "\n",
            )
        else:
            delta = m / ((np.exp(epsilon / k) - 1) * n + m)
            print(
                "Warning automatic = False, the parameter given may not guarantee privacy"
            )
            print(
                "k*log((1-delta)* m/n*delta + 1) = ",
                k * np.log((1 - delta) * m / (n * delta) + 1),
                "epsilon =",
                epsilon,
                "Is epsilon private =",
                np.isclose(k * np.log((1 - delta) * m / (n * delta) + 1), epsilon),
                "\n",
            )
            print(
                epsilon,
                "-DP achieved:",
                np.isclose(k * np.log((1 - delta) * m / (n * delta) + 1), epsilon),
            )

        # Perform histogram estimation
        hist_estimator, rescaling_factors = histogram_estimator(
            X, adaptative=adaptative, method="smooth_L2"
        )

    if norm == "KS":

        m = n ** (d / (d + 6))
        if automatic:
            k = int(n ** (4 / (d + 6)))  # k must be int
            delta = m * k / (n * epsilon)
            print(
                "Parameters k, delta and m where chosen to meet privacy:\n",
                "k = ",
                k,
                "\n",
                "delta = ",
                delta,
                "\n",
                "m = ",
                m,
                "\n",
            )

        else:
            delta = m / ((np.exp(epsilon / k) - 1) * n + m)
            print(
                "Warning overwrite = False, the parameter given may not guarantee privacy "
            )
            print(
                "k*log((1-delta)* m/n*delta + 1) = ",
                k * np.log((1 - delta) * m / (n * delta) + 1),
                "epsilon =",
                epsilon,
                "Is epsilon private =",
                np.isclose(
                    k * np.log((1 - delta) * m / (n * delta) + 1), epsilon, atol=1e-6
                ),
                "\n",
            )

        # Perform histogram estimation
        hist_estimator, rescaling_factors = histogram_estimator(
            X, adaptative=adaptative, method="smooth_KS"
        )

    # Generate DP synthetic data by smoothing
    DP_hist_smoothed = smooth_histogram(hist_estimator, delta=delta)

    smoothed_synthetic_data = generate_data_from_hist(
        DP_hist_smoothed, k=k, rescaling_factor=rescaling_factors, shuffle=shuffle
    )
    print("delta used=", delta)
    return smoothed_synthetic_data


def generate_perturbated_data(X, k, epsilon, adaptative=True, shuffle=True):

    n, d = X.shape

    # Perform histogram estimation
    hist_estimator, rescaling_factors = histogram_estimator(X, adaptative=adaptative)

    # Generate DP synthetic data by perturbation
    DP_hist_perturbed = perturbed_histogram(hist_estimator, epsilon=epsilon, n=n)

    perturbed_synthetic_data = generate_data_from_hist(
        DP_hist_perturbed, k=k, rescaling_factor=rescaling_factors, shuffle=shuffle
    )

    return perturbed_synthetic_data


# %%


n = 5000
d = 2

mean = [0, 1]
covariance_matrix = [[1, 0.8], [0.8, 1]]

# Generate random sample
X = np.random.multivariate_normal(mean, covariance_matrix, size=n)

perturbated = generate_smooth_data(
    X, k=5000, epsilon=25, adaptative=True, norm="L2", automatic=False
)

# perturbated, hist = generate_perturbated_data(X, k=500, epsilon=0.4, adaptative=True)

plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(perturbated[:, 0], perturbated[:, 1], alpha=0.5)
# %%
np.isclose(0.20000000000048782, 0.21)
