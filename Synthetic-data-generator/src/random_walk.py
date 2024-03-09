# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from histogram_estimator import histogram_estimator, generate_data_from_hist


def get_i_k(j):
    """
    Calculate the values of i and k for the given j.

    Parameters:
        j (int): Index of the basis function.

    Returns:
        tuple: A tuple containing the values of i and k.
    """
    assert j >= 0, "Error: j should be positive"
    if j == 1:
        return (0, 1)
    elif j == 2:
        return (1, 1)
    else:
        j -= 1
        i = int(np.log2(j))
        k = j - 2**i
        return (i + 1, 2 * k + 1)


def random_walk_laplace(m):
    """
    Generate Laplace noise for a random walk.

    Parameters:
        m (int): Length of the random walk.

    Returns:
        array-like: Array of Laplace noise with scale parameter L + 2.
    """

    assert m >= 0, "Error: argument should be positive"

    L = int(np.log2(m))

    laplace_noise = np.random.laplace(loc=0, scale=L + 2, size=m)

    return laplace_noise


def phi_bar_i_k(i, k, t):
    """
    Compute the j-th Schauder basis function at time points t.

    Parameters:
        i (int): Index of the basis function.
        k (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.

    Returns:
        array-like: Values of the j-k-th Schauder basis function evaluated at time points t.
    """
    t = np.asarray(t)  # Ensure t is a numpy array

    # Compute the values of the j-k-th Schauder basis function
    phi_values = 1 - 2**i * np.abs(t - k / 2**i)

    # Set negative values to zero

    phi_values[phi_values < 0] = 0

    return phi_values


def phi_bar(j, t):
    """
    Compute the j-th Schauder basis function at time points t.

    Parameters:
        j (int): Index of the basis function. Converted via get_index(j) to tuple index
        t (float or array-like): Time points at which to evaluate the function.

    Returns:
        array-like: Values of the j-th Schauder basis function evaluated at time points t.
    """
    # Get the values of i and k for the given j
    i, k = get_i_k(j)

    # Call phi_bar_i_k function to compute the Schauder basis function
    return phi_bar_i_k(i, k, t)


def psi_bar_i_k(i, k, t):
    """
    Compute the derivative of the Schauder basis function at time points t.

    Parameters:
        i (int): Index i.
        k (int): Index k.
        t (array-like): Time points at which to evaluate the derivative.

    Returns:
        array-like: Values of the derivative of the Schauder basis function evaluated at time points t.
    """
    phi_values = phi_bar_i_k(i, k, t)

    if (i, k) == (0, 1):
        return np.ones(len(t))
    psi_values = np.zeros(len(t))

    # TODO Make this cleaner (works fine)

    psi_values[t - k / 2**i > 0] = -(2 ** (i - 1))
    psi_values[(t - k / 2**i <= 0)] = 2 ** (i - 1)
    psi_values[(phi_values == 0) & (1 - 2**i * np.abs(t - k / 2**i) == 0)] = -(
        2 ** (i - 1)
    )

    psi_values[(phi_values == 0) & (1 - 2**i * np.abs(t - k / 2**i) != 0)] = 0
    psi_values[(t - k / 2**i <= 0) & (1 - 2**i * np.abs(t - k / 2**i) == 0)] = 0

    return psi_values


def psi_bar_j(j, t):
    """
    Compute the derivative of the j-th Schauder basis function at time points t.

    Parameters:
        j (int): Index of the basis function.
        t (array-like): Time points at which to evaluate the derivative.

    Returns:
        array-like: Values of the derivative of the j-th Schauder basis function evaluated at time points t.
    """
    # Get the values of i and k for the given j
    i, k = get_i_k(j)

    # Call psi_bar_i_k function to compute the Schauder basis function
    return psi_bar_i_k(i, k, t)


def precompute_basis_index(m):
    """
    Precompute basis functions for given length m.

    Parameters:
        m (int): Length of the basis function array.

    Returns:
        list: List of tuples containing the precomputed i and k index values.
    """

    basis_index = []
    for j in range(1, m + 1):
        i, k = get_i_k(j)
        basis_index.append((i, k))
    return basis_index


def precompute_psi_bar(m, basis_index, t_values):
    """
    Precompute psi_bar values for given basis functions index and time values.

    Parameters:
        m (int): Length of the psi_bar array = length of noise we are going to add.
        basis_index (list): List of tuples containing indexs i and k values.
        t_values (array-like): Array of time values.

    Returns:
        array-like: Array of precomputed psi_bar values.
    """

    psi_values = np.zeros((m, len(t_values)))
    for j in range(1, m + 1):
        i, k = basis_index[j - 1]
        psi_values[j - 1] = psi_bar_i_k(i, k, t_values)
    return psi_values / m


def super_regular_noise(m, n, epsilon):
    """
    Generate super-regular noise with differential privacy
    using precomputed functions

    Parameters:
        m (int): Length of the noise vector.
        n (int); size of data
        epsilon (float): Privacy parameter.

    Returns:
        array-like: Array of differentially private super-regular noise.
    """
    basis_index = precompute_basis_index(m)
    t_values = np.arange(1, m + 1) / m
    psi_bar_values = precompute_psi_bar(m, basis_index, t_values)

    privacy_factor = 2 / (epsilon * n)
    laplace_noise = random_walk_laplace(m)
    super_regular_noise = np.dot(laplace_noise, psi_bar_values)
    private_super_regular_noise = privacy_factor * super_regular_noise

    """print(
        t_values,
        "####################PSI",
        len(psi_bar_values),
        psi_bar_values,
        "\n",
        "####################Laplace",
        len(laplace_noise),
        laplace_noise,
    )"""

    return private_super_regular_noise


def minimize_signed_measure(omega, nu):
    """
    Minimize the given minimization problem to transform the DP-signed measure into
    DP-probability measure, to generate synthetic data.

    Parameters:
        omega (array-like): Array of values representing the events ω.
        nu (array-like): Array of values representing the signed measure of events.
        nu_prob (array-like): Array of values representing the probability of events.
            (That we are trying to estimate)

    Returns:
        float: DP-probability measure as close as possible so nu w.r.t to Wasserstein metric
    """
    m = len(omega)

    def objective_function(nu_b):
        """
        Objective function to minimize.

        Parameters:
            nu_b (array-like): Array of values representing estimated prob of events

        Returns:
            float: The value of the objective function.
        """
        obj = 0
        for k in range(m):
            if k == m - 1:
                obj += (1 - omega[k]) * np.abs(np.sum(nu_b[:k]) - np.sum(nu[:k]))
            else:
                obj += (omega[k + 1] - omega[k]) * np.abs(
                    np.sum(nu_b[:k]) - np.sum(nu[:k])
                )
        return obj

    # Constraint functions
    def positivity(x):
        """
        Constraint: nu(ωi) >= 0
        """
        return x

    def sum_to_one(x):
        """
        Constraint: Sum of nu(ωi) = 1
        """
        return np.sum(x) - 1

    # Initial guess for νb
    nu_prob_0 = np.ones(m) / m

    # Constraints
    constraints = [
        {"type": "ineq", "fun": positivity},
        {"type": "eq", "fun": sum_to_one},
    ]

    # Bounds for νb(ωi)
    bounds = [(0, 1)] * m

    # Solve the optimization problem
    result = minimize(
        objective_function,
        nu_prob_0,
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)


def private_measure_via_random_walk(X, epsilon, adaptative=True, display=False):

    # Calculate the dimensions of the data
    n, d = X.shape

    assert d == 1, "Error: d>1 is not implemented for this method at the moment"

    if adaptative == True:
        h = n ** (-1 / (2 + d))
        h_inverse = 1 / h
        bin_per_axis = int(np.ceil(h_inverse))
    else:
        bin_per_axis = n

    histogram, rescaling_factors = histogram_estimator(
        X, h=1 / n, adaptative=adaptative
    )

    privacy_noise = super_regular_noise(bin_per_axis, n, epsilon)

    noisy_histogram = histogram + privacy_noise

    omega = np.arange(0, bin_per_axis) / bin_per_axis

    prob_measure = minimize_signed_measure(omega, noisy_histogram)

    if display:
        print("histogram", histogram)

        print("Noisy hist", noisy_histogram)
        bins_edges = np.arange(0, bin_per_axis) / bin_per_axis
        plt.scatter(bins_edges, histogram, label="True histogram")
        # plt.scatter(bins_edges, noisy_histogram, label="Noisy signed measure")
        plt.scatter(bins_edges, prob_measure, label="Final histogram prob")
        plt.legend()
        plt.show()

    return prob_measure, rescaling_factors


# %%

n = 500
d = 1

X = np.random.normal(loc=0, scale=1, size=(n, d))

hist, rescale = private_measure_via_random_walk(
    X, epsilon=0.4, display=True, adaptative=True
)

# %%
