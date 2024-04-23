from histogram_DP_mechanism import (
    generate_smooth_data,
    generate_perturbed_data,
)
from super_regular_noise import generate_super_regular_noise_data
from linear_stat_fit import generate_auto_linear_stat_fit_data


def generate_data(X, size, epsilon, method, shuffle=True, rescaling=True, verbose=0):
    if method == "perturbed":
        return generate_perturbed_data(
            X, size, epsilon, shuffle=shuffle, rescaling=rescaling, verbose=verbose
        )
    elif method == "smooth":
        return generate_smooth_data(
            X,
            size,
            epsilon,
            shuffle=shuffle,
            automatic=True,
            rescaling=rescaling,
            verbose=verbose,
        )
    elif method == "linear_stat_fit_grid":
        return generate_auto_linear_stat_fit_data(
            X, size, epsilon, method="classic", shuffle=shuffle, rescaling=rescaling
        )
    elif method == "linear_stat_fit_reg":
        return generate_auto_linear_stat_fit_data(
            X, size, epsilon, method="linear_reg", shuffle=shuffle, rescaling=rescaling
        )
    elif method == "super_regular_noise":
        return generate_super_regular_noise_data(
            X, size, epsilon, shuffle=shuffle, rescaling=rescaling, verbose=verbose
        )
    else:
        raise ValueError("Unknown method")
