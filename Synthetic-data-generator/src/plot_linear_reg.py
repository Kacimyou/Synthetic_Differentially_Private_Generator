# %%
import numpy as np
import matplotlib.pyplot as plt
from histogram_DP_mechanism import generate_smooth_data, generate_perturbated_data
from super_regular_noise import generate_super_regular_noise_data
from linear_stat_fit import generate_auto_linear_stat_fit_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# %%


def generate_data(X, size, epsilon, method):
    if method == "perturbated":
        return generate_perturbated_data(X, size, epsilon)
    elif method == "smooth":
        return generate_smooth_data(X, size, epsilon)
    elif method == "linear_stat_fit":
        return generate_auto_linear_stat_fit_data(X, size, epsilon)
    elif method == "super_regular_noise":
        return generate_super_regular_noise_data(X, size, epsilon)
    else:
        raise ValueError("Unknown method")


def run_experiment(N, epsilon, methods, num_trials=10):

    sizes = np.logspace(np.log10(100), np.log10(N), num=10, dtype=int)

    for method in methods:
        mean_r2_scores = []
        variances = []
        for size in sizes:
            r2_scores = []
            for _ in range(num_trials):
                X = np.random.multivariate_normal(
                    mean=[0, 1], cov=[[1, 0.9], [0.9, 1]], size=size
                )
                private_data = generate_data(X, size, epsilon, method)

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
            variance = np.var(r2_scores)
            mean_r2_scores.append(mean_r2)
            variances.append(variance)

        # Plot mean R2 scores with variance for each method
        plt.errorbar(
            sizes,
            mean_r2_scores,
            yerr=variances,
            label=f"Method: {method}, Epsilon={epsilon}",
            fmt="-o",
        )

    plt.xlabel("Size of Initial Database")
    plt.ylabel("Mean R2 Score")
    plt.grid()
    plt.title("Mean R2 Score vs Size of Initial Database with Variance")
    plt.legend()
    plt.show()


# Example usage:
# methods = ["super_regular_noise","perturbated"]
methods = ["linear_stat_fit"]
run_experiment(2000, 2, methods, num_trials=1)

# %%
