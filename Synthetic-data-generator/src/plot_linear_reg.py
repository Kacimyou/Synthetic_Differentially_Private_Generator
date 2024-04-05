# %%
import numpy as np
import matplotlib.pyplot as plt
from histogram_DP_mechanism import generate_smooth_data, generate_perturbated_data
from super_regular_noise import generate_super_regular_noise_data
from linear_stat_fit import generate_auto_linear_stat_fit_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats

# %%


def generate_data(X, size, epsilon, method):
    if method == "perturbated":
        return generate_perturbated_data(X, size, epsilon)
    elif method == "smooth":
        return generate_smooth_data(X, size, epsilon)
    elif method == "linear_stat_fit":
        return generate_auto_linear_stat_fit_data(X, size, epsilon, method="grid_noid")
    elif method == "super_regular_noise":
        return generate_super_regular_noise_data(X, size, epsilon)
    else:
        raise ValueError("Unknown method")


def run_experiment(N, epsilon, methods, num_trials=10):

    sizes = np.logspace(np.log10(100), np.log10(N), num=10, dtype=int)

    for method in methods:
        mean_r2_scores = []
        stds = []
        for size in sizes:
            r2_scores = []
            for _ in range(num_trials):
                data = np.random.multivariate_normal(
                    mean=[0, 1], cov=[[1, 0.9], [0.9, 1]], size=size
                )
                private_data = generate_data(data, size, epsilon, method)

                x = private_data[:, 0].reshape(-1, 1)
                y = private_data[:, 1]

                # Perform linear regression
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)

                # Compute R2 score
                r2 = r2_score(y, y_pred)
                r2_scores.append(r2)

            # Calculate mean R2 score and variance
            mean_r2 = np.mean(r2_scores)
            std = np.std(r2_scores)
            mean_r2_scores.append(mean_r2)
            stds.append(std)

        stds = np.array(stds) * (scipy.stats.norm.ppf(0.975) / np.sqrt(num_trials))
        # Plot mean R2 scores with variance for each method
        plt.errorbar(
            sizes,
            mean_r2_scores,
            yerr=stds,
            label=f"Method: {method}, Epsilon={epsilon}",
            fmt="-o",
            linestyle="-",
            capsize=3,  # Length of the error bar caps
            capthick=0.5,  # Thickness of the error bar caps
            barsabove=True,  # Error bars are drawn above the markers
        )

    plt.xlabel("Size of Initial Database")
    plt.ylabel("Mean R2 Score")
    plt.grid()
    plt.title("Mean R2 Score vs Size of Initial Database with Variance")
    plt.legend()
    plt.show()


# Example usage:
# methods = ["super_regular_noise","perturbated"]
methods = ["perturbated"]
run_experiment(1000, 0.2, methods, num_trials=100)

# %%
