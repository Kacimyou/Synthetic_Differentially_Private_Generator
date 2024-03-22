# %%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import get_histogram_indices, generate_grid_points, scale, sample_from_grid


def is_in_bin(point, bin_index, bin_width):
    """
    Check if a given index corresponds to the bin 0.

    Parameters:
        point (numpy.ndarray): Point to test.
        bin_index (tuple): Index of the bin in the hypercube.
        bin_width (float): Width of each bin in the hypercube.

    Returns:
        int: 1 if the input is the bin, otherwise 0.
    """
    assert len(bin_index) == len(
        list(point)
    ), "Dimension of bin index and point must be equal."

    bin_center = [(i + 0.5) * bin_width for i in bin_index]  # Compute bin center
    distance = max(
        abs(point[i] - bin_center[i]) for i in range(len(bin_index))
    )  # Compute infinity norm distance
    if (
        distance <= bin_width / 2
    ):  # Check if the distance is within half of the bin width
        return 1
    return 0


def generate_bin_check_functions(m, d):
    """
    Generate a list of lambda functions to check if a point falls into specific bins.

    Parameters:
        bin_width (float): Width of each bin in the hypercube.
        bin_indices (list of tuples): List of bin indices.

    Returns:
        list: List of lambda functions to check if a point falls into specific bins.


    """
    bin_indices = get_histogram_indices(m, d)
    bin_width = 1 / m

    return [
        lambda x, bin_index=bin_index, bin_width=bin_width: is_in_bin(
            x, bin_index, bin_width
        )
        for bin_index in bin_indices
    ]


def fit_linear_stat(X, test_functions, reduced_space, noise):
    """
    Fit linear statistics of the true data on the reduced space. Output
    weight such that reduced space encapture as much information of the true data as possible.
    Add some noise in the optimization process to maintain privacy.

    Parameters:
        X (array_like): True data points.
        test_functions (list of callable): List of functions to compute statistics.
        reduced_space (array_like): Reduced space for statistical fitting.
        noise (array_like): Noise added to the true statistics to achieve differential privacy.

    Returns:
        numpy.ndarray: Fitted coefficients for the linear statistics.
    """

    m = len(reduced_space)
    X_scaled, rescaling_factors = scale(X)

    # Precompute true statistics
    true_stats = np.array([np.mean([f(x) for x in X_scaled]) for f in test_functions])
    reduced_space_stats = np.array(
        [[f(z) for z in reduced_space] for f in test_functions]
    )
    # print("true_stat\n", true_stats)

    def objective(h):
        linear_stats = np.dot(reduced_space_stats, h)
        return np.max(np.abs(linear_stats - true_stats - noise))

    def constraint(h):
        return np.sum(h) - 1

    initial_guess = np.ones(m) / m
    bounds = [(0, 1) for _ in range(m)]  # Ensure densities are non-negative
    constraints = [{"type": "eq", "fun": constraint}]

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    print(result.success)
    return result.x, rescaling_factors


def sample_from_linear_stat_density(
    linear_stat_density, reduced_space, k, d, rescaling_factor, shuffle=True
):
    """
    Generate synthetic data points by bootstrapping from the reduced space.

    Parameters:
        linear_stat_density (array_like): Reweighted density values for each point in the reduced space.
        reduced_space (list or array_like): Reduced space from which to sample data points.
        k (int): Number of synthetic data points to generate.
        d (int): Dimension of the initial data.

    Returns:
        array_like: Synthetic data points sampled from the reduced space.
    """

    # Normalize reweighted density, prevent from rounding error
    linear_stat_density /= np.sum(linear_stat_density)

    # Choose indices from the reduced space based on reweighted density
    chosen_indices = np.random.choice(len(reduced_space), size=k, p=linear_stat_density)

    # Retrieve the corresponding points from the reduced space
    synthetic_data = [reduced_space[index] for index in chosen_indices]

    if shuffle:

        noise = np.random.uniform(-1, 1, size=(len(synthetic_data), d)) / len(
            linear_stat_density
        ) ** (1 / d)
        print(len(linear_stat_density))

        synthetic_data += noise

    return np.array(synthetic_data)


def get_sigma_from_epsilon(epsilon, n, F, gamma=0.025):

    delta = 2 * F * np.log(F / gamma) / (n * epsilon)

    sigma = delta / np.log(F / gamma)

    print("sigma=", sigma, "delta", delta)
    return sigma


def generate_linear_stat_fit_data(
    X, test_functions, reduced_space, epsilon, k, shuffle=True
):
    """
    Generate private synthetic data points using linear statistical fitting.

    Parameters:
        X (array_like): True data points.
        test_functions (list of callable): List of functions to compute statistics.
        reduced_space (array_like): Reduced space from which to sample data points.
        epsilon (float): privacy budget.
        k (int): Number of synthetic data points to generate.

    Returns:
        numpy.ndarray: Synthetic data points sampled from the reduced space.
    """
    n, d = X.shape

    F = len(test_functions)
    sigma = get_sigma_from_epsilon(epsilon=epsilon, n=n, F=F)

    noise = np.random.laplace(scale=sigma, size=len(test_functions))
    linear_stat_density, rescaling_factors = fit_linear_stat(
        X, test_functions, reduced_space, noise
    )

    print("result of linear_stat_fit \n", np.round(linear_stat_density, decimals=4))
    synthetic_data = sample_from_linear_stat_density(
        linear_stat_density, reduced_space, k, d, rescaling_factors, shuffle=shuffle
    )
    print(rescaling_factors)
    return synthetic_data


def generate_auto_linear_stat_fit_data(
    X, epsilon, k, method="classic", shuffle=True, gamma=0.025
):
    """
    Generate private synthetic data points using linear statistical fitting.

    Parameters:
        X (array_like): True data points.
        epsilon (float): privacy budget.
        k (int): Number of synthetic data points to generate.
        method (string): indicate the method automatically generate test_functions and reduced space
        shuffle (bool): Whether to shuffle the reduced space.

    Returns:
        numpy.ndarray: Synthetic data points sampled from the reduced space.
    """
    n, d = X.shape
    assert method in ["classic", "grid_noid"], "Error: method not defined"

    m = int(
        np.ceil(n ** (1 / (2 + d)))
    )  # m is calculated with the formula of histogram estimation
    # print("m=", m)
    if method == "classic":

        # Generate test functions
        test_functions = generate_bin_check_functions(m, d)

        # Generate reduced space
        reduced_space = sample_from_grid(m, d, 3 * m)
        # print("reduced_space", reduced_space)

    if method == "grid_noid":

        # Generate test functions
        test_functions = generate_bin_check_functions(m, d)

        # Generate reduced space
        reduced_space = generate_grid_points(m, d)
        # print("reduced_space", reduced_space)

    F = len(test_functions)
    sigma = get_sigma_from_epsilon(epsilon=epsilon, n=n, F=F, gamma=gamma)

    # Generate noise
    noise = np.random.laplace(scale=sigma, size=len(test_functions))

    # Fit linear statistical model
    linear_stat_density, rescaling_factors = fit_linear_stat(
        X, test_functions, reduced_space, noise
    )

    # Sample synthetic data points based on linear statistical model
    synthetic_data = sample_from_linear_stat_density(
        linear_stat_density, reduced_space, k, d, rescaling_factors, shuffle=shuffle
    )

    return synthetic_data


# %%
# Example usage with multiple test functions, what we do in practice:
n = 5000
d = 2
epsilon = 0.1
k = 3000  # Example number of synthetic data points

mean = [0, 1]
covariance_matrix = [[1, 0.8], [0.8, 1]]

# Generate random sample
X = np.random.multivariate_normal(mean, covariance_matrix, size=n)


synthetic_data = generate_auto_linear_stat_fit_data(X, epsilon, k, method="grid_noid")

plt.scatter(scale(X[:, 0])[0], scale(X[:, 1])[0], alpha=0.5)
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.5)

# %%

plt.scatter(np.linspace(0, 1, 1000), np.sort(synthetic_data[:, 0]))

# %%
get_sigma_from_epsilon(2, 10000, 100)
