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
    if j == 1:
        return (0, 1)
    elif j == 2:
        return (1, 1)
    else:
        j -= 1
        i = int(np.log2(j))
        k = j - 2**i
        return (i + 1, 2 * k + 1)


def random_walk_laplace(n):
    """
    Generate Laplace noise for a random walk.

    Parameters:
        n (int): Length of the random walk.

    Returns:
        array-like: Array of Laplace noise with scale parameter L + 2.
    """

    L = int(np.log2(n))

    laplace_noise = np.random.laplace(loc=0, scale=L + 2, size=n)

    return laplace_noise


def phi_bar_i_k(i, k, t):
    """
    Compute the j-th Schauder basis function at time points t.

    Parameters:
        j (int): Index of the basis function.
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
        j (int): Index of the basis function.
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


def precompute_basis_index(n):
    """
    Precompute basis functions for given length n.

    Parameters:
        n (int): Length of the basis function array.

    Returns:
        list: List of tuples containing the precomputed i and k index values.
    """

    basis_index = []
    for j in range(1, n + 1):
        i, k = get_i_k(j)
        basis_index.append((i, k))
    return basis_index


def precompute_psi_bar(n, basis_index, t_values):
    """
    Precompute psi_bar values for given basis functions index and time values.

    Parameters:
        n (int): Length of the psi_bar array.
        basis_index (list): List of tuples containing indexs i and k values.
        t_values (array-like): Array of time values.

    Returns:
        array-like: Array of precomputed psi_bar values.
    """

    psi_values = np.zeros((n, len(t_values)))
    for j in range(1, n + 1):
        i, k = basis_index[j - 1]
        psi_values[j - 1] = psi_bar_i_k(i, k, t_values)
    return psi_values / n


def super_regular_noise(n, epsilon):
    """
    Generate super-regular noise with differential privacy
    using precomputed functions

    Parameters:
        n (int): Length of the noise vector.
        epsilon (float): Privacy parameter.

    Returns:
        array-like: Array of differentially private super-regular noise.
    """
    basis_index = precompute_basis_index(n)
    t_values = np.arange(1, n + 1) / n
    psi_bar_values = precompute_psi_bar(n, basis_index, t_values)

    privacy_factor = 2 / (epsilon * n)
    laplace_noise = random_walk_laplace(n)
    super_regular_noise = np.dot(laplace_noise, psi_bar_values)
    private_super_regular_noise = privacy_factor * super_regular_noise

    print(
        t_values,
        "####################PSI",
        len(psi_bar_values),
        psi_bar_values,
        "\n",
        "####################Laplace",
        len(laplace_noise),
        laplace_noise,
    )
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
    n = len(omega)

    def objective_function(nu_b):
        """
        Objective function to minimize.

        Parameters:
            nu_b (array-like): Array of values representing estimated prob of events

        Returns:
            float: The value of the objective function.
        """
        obj = 0
        for k in range(n):
            if k == n - 1:
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
    nu_prob_0 = np.ones(n) / n

    # Constraints
    constraints = [
        {"type": "ineq", "fun": positivity},
        {"type": "eq", "fun": sum_to_one},
    ]

    # Bounds for νb(ωi)
    bounds = [(0, 1)] * n

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

    histogram, rescaling_factors = histogram_estimator(X, adaptative=True)

    if adaptative == True:
        h = n ** (-1 / (2 + d))
        h_inverse = 1 / h
        bin_per_axis = int(np.ceil(h_inverse))
    else:
        bin_per_axis = n

    privacy_noise = super_regular_noise(bin_per_axis, epsilon) / bin_per_axis

    noisy_histogram = histogram + privacy_noise

    omega = np.arange(0, bin_per_axis) / bin_per_axis

    prob_measure = minimize_signed_measure(omega, noisy_histogram)

    if display:
        print("histogram", histogram)

        print("Noisy hist", noisy_histogram)
        plt.scatter(np.arange(0, bin_per_axis) / bin_per_axis, histogram)
        # plt.scatter(np.arange(0, bin_per_axis) / bin_per_axis, noisy_histogram)
        plt.scatter(np.arange(0, bin_per_axis) / bin_per_axis, prob_measure)

    return prob_measure, rescaling_factors


# %%
super_regular_noise(4, 1)

# %%

n = 10000
d = 1
omega = np.arange(0, n) / n
nu = [
    -0.03681204,
    0.06048189,
    -0.0361927,
    -0.05452837,
    -0.0381622,
    0.00469401,
    0.10321175,
    -0.06149333,
    0.047348,
    0.10465018,
    -0.03489208,
    0.01388376,
    0.10618425,
    -0.05106611,
    -0.00062602,
    -0.0263301,
    0.07857817,
    -0.10026255,
    -0.03543559,
    -0.0347201,
    0.13738909,
    -0.1846867,
    -0.02046825,
    0.08659036,
    0.01295036,
    0.0722068,
    0.12607191,
    -0.20018651,
    -0.08609094,
    0.21247491,
    0.12300099,
    -0.01543522,
    0.18092333,
    -0.05010617,
    0.12507875,
    0.05338238,
    0.07057642,
    0.14803861,
    -0.08378199,
    0.19660702,
    0.06867085,
    0.28720526,
    0.03612286,
    -0.13231207,
    0.03952573,
    0.28438344,
    0.24392974,
    0.16877611,
    0.12789762,
    0.04206525,
    -0.26262078,
    -0.11287849,
    0.08236662,
    -0.15812994,
    -0.4107314,
    0.07624718,
    0.19236165,
    -0.18038165,
    -0.01727428,
    0.16153415,
    0.04801131,
    -0.0438784,
    -0.10171691,
    0.14776543,
    -0.2156104,
    0.02925715,
    0.15257545,
    0.03944996,
    -0.07377091,
    0.32002718,
    0.04505783,
    0.03824109,
    0.04242793,
    0.00627655,
    0.23013874,
    -0.00713244,
    0.00230287,
    -0.030503,
    0.1138652,
    -0.18145182,
    -0.19894155,
    -0.11676812,
    -0.12269779,
    -0.06220849,
    -0.08589807,
    0.0929764,
    0.1150404,
    0.10371585,
    -0.10641246,
    -0.02227546,
    0.12265577,
    0.17164574,
    -0.23153962,
    -0.10283746,
    0.23473566,
    -0.17086455,
    -0.17134639,
    -0.07726667,
    -0.0596133,
    -0.04123565,
]

# minimize_signed_measure(omega, nu)

# %%
X = np.random.normal(loc=0, scale=1, size=(n, d))
hist, rescale = private_measure_via_random_walk(X, epsilon=0.70, display=True)
print(hist)
# %%
plt.hist(hist)
# %%
