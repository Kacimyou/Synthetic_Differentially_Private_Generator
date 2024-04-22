# %%
import numpy as np
import itertools


def generate_grid_points(m, d):
    """
    Generate 1/m-net of [0,1]**d.

    Parameters:
        m (int): Number of points in each dimension.
        d (int): Dimensionality of the space.

    Returns:
        numpy.ndarray: Array of grid points with shape (m^d, d).
    """
    # Generate grid points for each dimension
    grid_points = [np.linspace(1 / (2 * m), 1 - 1 / (2 * m), m) for _ in range(d)]
    # Take the Cartesian product of grid points
    points = np.array(list(itertools.product(*grid_points)))
    return points


def sample_from_grid(m, d, k):
    """
    Generate k independent samples from the grid points.

    Parameters:
        m (int): Number of points in each dimension.
        d (int): Dimensionality of the space.
        k (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Array of sampled points with shape (k, d).
    """
    grid_points = generate_grid_points(m, d)
    indices = np.random.choice(len(grid_points), k, replace=True)
    return grid_points[indices]


def get_histogram_indices(m, d):
    """
    Generate all indices of a d multidimensional histogram.

    Parameters:
        m (int): Number of bins in each dimension.
        d (int): Dimensionality of the space.

    Returns:
        list: List of all histogram indices.
    """
    index_ranges = [range(m) for _ in range(d)]
    return list(itertools.product(*index_ranges))


def scale(X):
    """
    Scales the input data `X` along each feature axis to the range [0, 1].

    Parameters:
        X (array-like): Input data of shape (n_samples, n_features).

    Returns:
        X_scaled (numpy.ndarray): Scaled data with the same shape as `X`.
        rescaling_factors (numpy.ndarray): An array containing the minimum and maximum values
            along each feature axis, used for rescaling. Shape is (2, n_features).
    """
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    rescaling_factors = np.vstack((min_values, max_values))

    X_scaled = (X - rescaling_factors[0]) / (
        rescaling_factors[1] - rescaling_factors[0]
    )

    return X_scaled, rescaling_factors


def rescale(X, rescaling_factors):
    """
    Rescales the input data `X` based on the given rescaling factors.

    Parameters:
        X (array-like): Scaled data of shape (n_samples, n_features).
        rescaling_factors (numpy.ndarray): An array containing the minimum and maximum values
            along each feature axis, used for rescaling. Shape is (2, n_features).

    Returns:
        X_rescaled (numpy.ndarray): Rescaled data with the same shape as `X`.
    """
    min_values = rescaling_factors[0]
    max_values = rescaling_factors[1]

    X_rescaled = X * (max_values - min_values) + min_values

    return X_rescaled


# # Test
# size = 500
# d = 30
# X = np.random.randn(size, d)
# print("X",'\n',X,"\n")

# # Scaling
# X_scale, r = scale(X)
# print("X_scale",'\n',X_scale,'\n', r, "\n")

# # Rescaling
# X_rescale = rescale(X_scale, r)
# print("X_rescale",'\n',X_rescale)

# # Check if X_rescale == X
# print("X_rescale == X:", np.allclose(X_rescale, X))
