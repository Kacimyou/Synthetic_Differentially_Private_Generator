import numpy as np
from scipy.optimize import minimize

def laplace_noise(scale, size):
    return np.random.laplace(scale=scale, size=size)

def reweight_density(true_data, test_functions, reduced_space, sigma, noise):
    n = len(true_data)
    m = len(reduced_space)
    
    def objective(h):
        linear_stats = np.array([np.sum([f(z) * h[idx] for idx, z in enumerate(reduced_space)]) for f in test_functions])
        true_stats = np.array([np.mean([f(x) for x in true_data]) for f in test_functions])
        return np.max(np.abs(linear_stats - true_stats - noise))

    def constraint(h):
        return np.sum(h) - 1

    initial_guess = np.ones(len(reduced_space)) / len(reduced_space)
    bounds = [(0, 1) for _ in range(len(reduced_space))]  # Ensure densities are non-negative
    constraints = [{'type': 'eq', 'fun': constraint}]

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    print(result.x, np.sum(result.x))
    return result.x

def bootstrap_data(reweighted_density, reduced_space, k):
    return np.random.choice(reduced_space, size=k, p=reweighted_density)

def private_synthetic_data(true_data, test_functions, reduced_space, sigma, k):
    noise = laplace_noise(sigma, len(test_functions))
    reweighted_density = reweight_density(true_data, test_functions, reduced_space, sigma, noise)
    synthetic_data = bootstrap_data(reweighted_density, reduced_space, k)
    return synthetic_data

# Example usage:
true_data = np.random.normal(loc=5, scale=1, size=100)  # Example true data
test_functions = [lambda x: x, lambda x : x**2-
                  
                  +]  # Example test functions
reduced_space = np.linspace(3, 6, 10)  # Example reduced space
sigma = 0.001  # Example noise parameter
k = 100  # Example number of synthetic data points
synthetic_data = private_synthetic_data(true_data, test_functions, reduced_space, sigma, k)
