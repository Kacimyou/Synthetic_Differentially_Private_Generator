# %%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import get_histogram_indices, generate_grid_points


def laplace_noise(scale, size):
    return np.random.laplace(scale=scale, size=size)


def reweight_density(true_data, test_functions, reduced_space, sigma, noise):
    n = len(true_data)
    m = len(reduced_space)

    # Precompute true statistics
    true_stats = np.array([np.mean([f(x) for x in true_data]) for f in test_functions])
    reduced_space_stats = np.array(
        [[f(z) for z in reduced_space] for f in test_functions]
    )
    print("true_stat", true_stats, len(true_stats))

    def objective(h):
        linear_stats = np.dot(reduced_space_stats, h)
        return np.max(np.abs(linear_stats - true_stats - noise))

    def constraint(h):
        return np.sum(h) - 1

    initial_guess = np.ones(len(reduced_space)) / len(reduced_space)
    bounds = [
        (0, 1) for _ in range(len(reduced_space))
    ]  # Ensure densities are non-negative
    constraints = [{"type": "eq", "fun": constraint}]

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x


def bootstrap_data(reweighted_density, reduced_space, k):
    """
    Generate synthetic data points by bootstrapping from the reduced space.

    Parameters:
        reweighted_density (array_like): Reweighted density values for each point in the reduced space.
        reduced_space (list or array_like): Reduced space from which to sample data points.
        k (int): Number of synthetic data points to generate.

    Returns:
        array_like: Synthetic data points sampled from the reduced space.
    """
    # Normalize reweighted density
    reweighted_density /= np.sum(reweighted_density)

    # Choose indices from the reduced space based on reweighted density
    chosen_indices = np.random.choice(len(reduced_space), size=k, p=reweighted_density)

    # Retrieve the corresponding points from the reduced space
    synthetic_data = [reduced_space[index] for index in chosen_indices]

    return np.array(synthetic_data)


def private_synthetic_data(true_data, test_functions, reduced_space, sigma, k):
    noise = laplace_noise(sigma, len(test_functions))
    reweighted_density = reweight_density(
        true_data, test_functions, reduced_space, sigma, noise
    )

    # print(reweighted_density)
    synthetic_data = bootstrap_data(reweighted_density, reduced_space, k)
    return synthetic_data


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
        point
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

    return [lambda x: is_in_bin(x, bin_index, bin_width) for bin_index in bin_indices]


# %%
# Example usage with 1 test function (useless):
true_data = np.random.normal(loc=0, scale=1, size=1000)  # Example true data
test_functions = [lambda x: x]  # Example test functions
reduced_space = np.linspace(-1, 1, 20)  # Example reduced space
sigma = 0.001  # Example noise parameter
k = 100  # Example number of synthetic data points
synthetic_data = private_synthetic_data(
    true_data, test_functions, reduced_space, sigma, k
)


plt.hist(synthetic_data)
# %%
# Example 2D histogram parameters
num_bins = 10
bin_width = 0.1

# Index of the bin to check (bin 0 in this case)
bin_index = (0, 0)

# Point to test
point = np.array([0.05, 0.05])  # Point lies within the bin 0

# Check if the point falls into the bin 0
result = is_in_bin(point, bin_index, bin_width)

# %%
m = 3
d = 2

for i in range(m**d):
    print(generate_bin_check_functions(m, d)[i](point))

# %%
# Example usage with 1 test function (useless):
n = 100
d = 2
true_data = np.random.normal(loc=0, scale=1, size=(n, d))  # Example true data
test_functions = generate_bin_check_functions(5, 2)

reduced_space = generate_grid_points(10, 2)  # Example reduced space
sigma = 0.001  # Example noise parameter
k = 100  # Example number of synthetic data points
synthetic_data = private_synthetic_data(
    true_data, test_functions, reduced_space, sigma, k
)


plt.hist(synthetic_data[:, 0])
# %%
