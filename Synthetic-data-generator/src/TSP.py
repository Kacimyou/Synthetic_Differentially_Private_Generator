# %%
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from histogram_estimator import histogram_estimator
from tqdm import tqdm
from utils import generate_grid_points


def is_neighbour(p1, p2, neighbour_distance, norm=2):
    """
    Check if two points are neighbours based on a given neighbour distance.

    Parameters:
        p1 (numpy.ndarray): First point.
        p2 (numpy.ndarray): Second point.
        neighbour_distance (float): Neighbour distance threshold.

    Returns:
        bool: True if the distance between p1 and p2 is less than or equal to neighbour_distance, False otherwise.
    """
    distance = np.linalg.norm(p1 - p2, ord=norm)
    return distance <= neighbour_distance


def tsp(points, norm=2):
    """
    Calculate the Traveling Salesman Problem (TSP) path given a finite metric space.

    Parameters:
        points (numpy.ndarray): Array of points in the space.

    Returns:
        list: TSP path as a list of points.
    """

    assert len(points) > 1, "Error: Length of points inferior or equal to 0."
    # neighbour_distance = np.linalg.norm(points[0]-points[1], ord = norm)

    G = nx.Graph()

    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            # if is_neighbour(p1, p2, neighbour_distance, norm):
            # TODO Check impact of norm
            weight = np.linalg.norm(p1 - p2, ord=norm)
            G.add_edge(i, j, weight=weight)

    # TODO Clean solving algorithm, precising what method is used.
    # tsp_path = nx.approximation.traveling_salesman_problem(G)
    tsp_path = nx.approximation.greedy_tsp(G)
    tsp_path.pop()
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


m = 4
points = generate_grid_points(m, 2)
# Set figure size
plt.gcf().set_size_inches(12, 6)


def plot_mapping(points, square=False, arrow=True):
    # Calculate TSP path
    tsp_path = tsp(points, norm=2)

    # Assign colors to original grid points with a gradient based on index
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

    # Plot for TSP path
    plt.subplot(1, 2, 1)

    # Determine the size of the square
    square_size = 1 / m - 10 ** (-3)
    if square:
        # Plot squares around each point with the same color as the point
        for i, point in enumerate(points):
            x, y = point
            square = plt.Rectangle(
                (x - square_size / 2, y - square_size / 2),
                square_size,
                square_size,
                color=colors[i],
                alpha=0.30,
            )
            plt.gca().add_patch(square)
        for i in range(len(points)):
            plt.axvline(points[i][0] - square_size / 2, color="grey", linestyle="--")
            plt.axvline(points[i][0] + square_size / 2, color="grey", linestyle="--")
            plt.axhline(points[i][1] - square_size / 2, color="grey", linestyle="--")
            plt.axhline(points[i][1] + square_size / 2, color="grey", linestyle="--")

        for i, point in enumerate(tsp_path):
            plt.annotate(
                r"$\mathcal{I}_{%s}$" % (i + 1),
                (point[0], point[1]),
                textcoords="offset points",
                xytext=(-15, -10),
                ha="center",
                fontsize=8,
            )

    if arrow:
        for i in range(len(tsp_path) - 1):

            dx = tsp_path[i + 1][0] - tsp_path[i][0]
            dy = tsp_path[i + 1][1] - tsp_path[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            plt.arrow(
                tsp_path[i][0],
                tsp_path[i][1],
                dx,
                dy,
                color="grey",
                head_width=0.02,
                head_length=0.03,
                width=0.0005,
            )
            plt.text(
                tsp_path[i][0] + dx / 2,
                tsp_path[i][1] + dy / 2,
                r"$\delta_{%s} = %.2f$" % (i + 1, distance),
                fontsize=8,
            )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Traveling Salesman Problem")
    plt.grid(False)

    # Annotate the points based on tsp_path
    for i, point in enumerate(tsp_path):
        plt.annotate(
            r"$\omega_{%s}$" % (i + 1),
            (point[0], point[1]),
            textcoords="offset points",
            xytext=(10, 3),
            ha="center",
            fontsize=8,
        )
        plt.scatter(tsp_path[i][0], tsp_path[i][1], marker="o", color=colors[i])

    # Compute cumulative distances
    cumulative_distances = [0]
    for i in range(1, len(tsp_path)):
        cumulative_distances.append(
            cumulative_distances[-1]
            + np.linalg.norm(np.array(tsp_path[i]) - np.array(tsp_path[i - 1]))
        )

    plt.subplot(1, 2, 2)

    plt.title("Mapping TSP Path to Line")
    plt.gca().get_yaxis().set_visible(False)  # Hide the y-axis
    plt.axhline(0, color="black", linestyle="--")  # Plot horizontal line at y = 0

    plt.scatter(
        cumulative_distances, [0] * len(tsp_path), marker="o", color=colors
    )  # Plotting the points with colors
    path_length = len(tsp_path)

    plt.title("Mapping TSP Path to Line")
    for i, point in enumerate(tsp_path):
        plt.annotate(
            r"$\omega_{%s}$" % (i + 1),
            (cumulative_distances[i], 0),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Annotate arrows with distance
    for i in range(len(tsp_path) - 1):
        dx = cumulative_distances[i + 1] - cumulative_distances[i]
        dy = 0
        distance = np.sqrt(dx**2 + dy**2)

        plt.text(
            cumulative_distances[i] + dx / 5,
            0.005 * (-1) ** i,
            r"$\delta_{%s} = %.2f$" % (i + 1, distance),
            fontsize=8,
        )

    plt.xlabel("Path Length")

    plt.tight_layout()
    plt.show()


uniform = np.random.uniform(0, 1, size=(10, 2))
# plot_mapping(points, square=False)

# %%

##Test that the application of both mapping gives back the original histogram
# n = 10000
# d = 3

# X = np.random.normal(loc=0, scale=1, size=(n, d))

# hist, rescale = histogram_estimator(X)


# one_dim_array, reverse_dict, shape, omega = from_multi_to_1D(hist)

# multi_dim = from_1D_to_multi(one_dim_array, reverse_dict, shape)
# print(multi_dim == hist)
