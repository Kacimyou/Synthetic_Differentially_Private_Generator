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
    grid_points = [np.linspace(0, 1, m) for _ in range(d)]
    # Take the Cartesian product of grid points
    points = np.array(list(itertools.product(*grid_points)))
    return points


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
