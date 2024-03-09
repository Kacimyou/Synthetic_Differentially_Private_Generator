# %%
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from histogram_estimator import histogram_estimator
from tqdm import tqdm


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


def tsp(points):
    """
    Calculate the Traveling Salesman Problem (TSP) path given a finite metric space.

    Parameters:
        points (numpy.ndarray): Array of points in the space.

    Returns:
        list: TSP path as a list of points.
    """
    G = nx.Graph()
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                # TODO Check if norm L2 should be used
                weight = np.linalg.norm(p1 - p2)
                G.add_edge(i, j, weight=weight)
    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
    tsp_path = [points[i] for i in tsp_path]
    return tsp_path


def mapping_bijection_dict(tsp_path, m, d):
    """
    Map the TSP path to the multidimensional histogram representation.
    Returns two dictionnary allowing the mapping of the d-dimensional hypercube to the
    [0, TSP(T)] interval.

    Parameters:
        tsp_path (list): TSP path as a list of points.
        m (int): Number of bins in each dimension.
        d (int): Dimensionality of the space.

    Returns:
        tuple: Forward and reverse mapping dictionaries.
    """
    h = 1 / m
    forward_dict = {}  # Forward mapping dictionary
    reverse_dict = {}  # Reverse mapping dictionary
    total_length = 0
    for i in range(0, len(tsp_path)):
        # Calculate Euclidean distance between consecutive points
        if i == 0:
            distance = 0
        else:
            distance = np.linalg.norm(np.array(tsp_path[i]) - np.array(tsp_path[i - 1]))

        bin_indices = tuple(
            (
                int(np.floor(tsp_path[i][j] / h)) - 1
                if np.floor(tsp_path[i][j]) == 1
                else int(np.floor(tsp_path[i][j] / h))
            )
            for j in range(0, d)
        )

        total_length += distance
        forward_dict[bin_indices] = (total_length, i)
        reverse_dict[i] = bin_indices  # Construct reverse dictionary
    return forward_dict, reverse_dict


def get_grid_dict(m, d):
    """
    Generate the mapping dictionnary for a 1/m-net of [0,1]**d.

    Parameters:
        m (int): Number of bins in each dimension.
        d (int): Dimensionality of the space.

    Returns:
        tuple: Forward and reverse mapping dictionaries.
    """

    points = generate_grid_points(m, d)

    tsp_path = tsp(points)

    foward_dict, reverse_dict = mapping_bijection_dict(tsp_path, m, d)

    return foward_dict, reverse_dict


def get_omega(foward_dict):
    return [e[0] for e in foward_dict.values()]


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


def from_multi_to_1D(histogram):
    """
    Convert a multidimensional histogram to a one-dimensional representation.
    This is used to add the super regular random walk in the multi-dim case.
    Privacy algorithm is defined for 1D only so we must convert it first.

    Parameters:
        histogram (numpy.ndarray): Multidimensional histogram.

    Returns:
        tuple: One-dimensional histogram, reverse mapping dictionary, shape of the original histogram,
        omega a list of value between 0 and TSP(T).
    """

    m = len(histogram)
    d = len(histogram.shape)
    foward_dict, reverse_dict = get_grid_dict(m, d)

    histogram_indices = get_histogram_indices(m, d)

    one_d_histogram = np.zeros(m**d)

    for index in histogram_indices:

        one_d_histogram[foward_dict.get(index)[1]] = histogram[index]

    omega = get_omega(foward_dict)

    return one_d_histogram, reverse_dict, (m, d), omega


def from_1D_to_multi(one_d_histogram, reverse_dict, shape):
    """
    Inverse function of from_multi_to_1D, use reverse_dict: output of from_multi_to_1D to
    map back the histogram to it's original shape.

    Parameters:
        one_d_histogram (numpy.ndarray): One-dimensional histogram.
        reverse_dict (dict): Reverse mapping dictionary.
        shape (tuple): Shape of the original histogram.

    Returns:
        numpy.ndarray: Multidimensional histogram.
    """

    m, d = shape

    multi_dim_histogram = np.zeros((m,) * d)

    for i in range(len(one_d_histogram)):

        index = reverse_dict[i]

        multi_dim_histogram[index] = one_d_histogram[i]

    return multi_dim_histogram


# %%

##Test that the application of both mapping gives back the original histogram
n = 1000
d = 1

X = np.random.normal(loc=0, scale=1, size=(n, d))

hist, rescale = histogram_estimator(X)

one_dim_array, reverse_dict, shape, omega = from_multi_to_1D(hist)

multi_dim = from_1D_to_multi(one_dim_array, reverse_dict, shape)
print(multi_dim == hist)
