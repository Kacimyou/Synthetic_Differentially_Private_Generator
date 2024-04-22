# %%
import numpy as np
import matplotlib.pyplot as plt
from histogram_DP_mechanism import (
    generate_smooth_data,
    generate_perturbated_data,
)
from super_regular_noise import generate_super_regular_noise_data
from linear_stat_fit import generate_auto_linear_stat_fit_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
import scipy.stats
from tqdm import tqdm


def generate_data(X, size, epsilon, method, shuffle=True, rescaling=True):
    if method == "perturbated":
        return generate_perturbated_data(
            X, size, epsilon, shuffle=shuffle, rescaling=rescaling
        )
    elif method == "smooth":
        return generate_smooth_data(
            X, size, epsilon, shuffle=shuffle, automatic=True, rescaling=rescaling
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
            X, size, epsilon, shuffle=shuffle, rescaling=rescaling
        )
    else:
        raise ValueError("Unknown method")


# %%


def run_experiment_R2(
    N,
    epsilon,
    methods,
    beta=[1],
    num_trials=10,
    d=1,
    num_step=10,
    noise_std=0.2,
    shuffle=True,
):

    sizes = np.logspace(np.log10(100), np.log10(N), num=num_step, dtype=int)

    for method in methods:
        print(f"## COMPUTING method = {method} ##")
        mean_r2_scores = []
        stds = []
        for size in tqdm(sizes):
            r2_scores = []
            for _ in tqdm(range(num_trials)):

                X = np.random.randn(size, d)
                y = (
                    np.dot(X, beta).reshape(size, 1)
                    + np.random.randn(size, 1) * noise_std
                )

                data = np.concatenate((X, y), axis=1)
                private_data = generate_data(
                    data, size, epsilon, method, shuffle=shuffle
                )

                x = private_data[:, 0].reshape(-1, 1)
                y = private_data[:, -1]

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
            label=f"{method}",
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


# Example usage:
# methods = ["super_regular_noise","perturbated","smooth", "linear_stat_fit_grid","linear_stat_fit_reg"]
methods = [
    "perturbated",
    "linear_stat_fit_reg",
]
run_experiment_R2(1000, 2, methods, num_trials=100, d=1, beta=[1], shuffle=False)


# %%
def run_experiment_beta(
    N, epsilon, methods, beta=[1], num_trials=10, d=1, noise_std=0.2, shuffle=True
):

    sizes = np.logspace(np.log10(100), np.log10(N), num=10, dtype=int)

    for method in methods:
        mse_scores = []
        stds = []
        for size in sizes:
            betas = []
            for _ in range(num_trials):

                assert (
                    len(beta) == d
                ), "Error: size of beta do not match number of features"
                X = np.random.randn(size, d)
                y = (
                    np.dot(X, beta).reshape(size, 1)
                    + np.random.randn(size, 1) * noise_std
                )

                data = np.concatenate((X, y), axis=1)
                private_data = generate_data(
                    data, size, epsilon, method, shuffle=shuffle
                )

                x = private_data[:, :-1]  # Features
                y_true = private_data[:, -1]  # Target

                # Perform linear regression
                model = LinearRegression()
                model.fit(x, y_true)
                beta_hat = model.coef_

                betas.append(beta_hat)

            # Calculate mean MSE and variance
            bias_beta = np.mean(np.array(betas) - np.array(beta))
            print(bias_beta)

            var_beta = np.var(betas)
            print(var_beta)
            std = np.std(betas)
            stds.append(std)
            mse_scores.append(bias_beta**2 + var_beta)

        stds = np.array(stds) * (scipy.stats.norm.ppf(0.975) / np.sqrt(num_trials))
        # Plot mean MSE scores with variance for each method
        plt.errorbar(
            sizes,
            mse_scores,
            yerr=stds,
            label=f"Method: {method}, Epsilon={epsilon}",
            fmt="-o",
            linestyle="-",
            capsize=3,  # Length of the error bar caps
            capthick=0.5,  # Thickness of the error bar caps
            barsabove=True,  # Error bars are drawn above the markers
        )

    plt.xlabel("Size of Initial Database")
    plt.ylabel("Mean Squared Error (MSE) of Coefficient Estimates")
    plt.grid()
    plt.title(
        "Mean Squared Error (MSE) of Coefficient Estimates vs Size of Initial Database with Variance"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


methods = ["smooth"]
run_experiment_beta(1000, 2, methods, num_trials=10, d=1, beta=[10], shuffle=False)


# %%


def plot_linear_reg_private(size, d, noise_std, beta, epsilon):
    X = np.random.randn(size, d)
    y = np.dot(X, beta).reshape(size, 1) + np.random.randn(size, 1) * noise_std
    data = np.concatenate((X, y), axis=1)

    # Assuming you have implemented the generate_smooth_data function correctly
    private_data = generate_data(
        data, size, epsilon, method="super_regular_noise", shuffle=True, rescaling=True
    )
    x_private = private_data[:, 0].reshape(-1, 1)
    y_private = private_data[:, -1]

    model = LinearRegression()
    model.fit(x_private, y_private)
    # Predictions
    y_pred = model.predict(x_private)

    # Fit another linear regression model to original data X and y
    original_model = LinearRegression()
    original_model.fit(X, y)
    y_pred_original = original_model.predict(X)

    # Plot original and private data
    plt.scatter(X, y, color="lightblue", alpha=0.5, label="Original Data")
    plt.scatter(
        private_data[:, 0],
        private_data[:, 1],
        color="pink",
        alpha=0.7,
        label="Private Data",
    )

    # Plot the estimated line from linear regression on private data
    plt.plot(
        x_private,
        y_pred,
        color="red",
        linestyle="dotted",
        linewidth=3,
        label=f'Private Regression beta_hat ={model.coef_}"',
    )

    # Plot the estimated line from linear regression on original data
    plt.plot(
        X,
        y_pred_original,
        color="blue",
        linewidth=3,
        linestyle="dotted",
        label=f'Original Regression beta_hat ={original_model.coef_}"',
    )

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Generate synthetic data
size = 10000
d = 1
noise_std = 10
beta = 20
epsilon = 0.5


plot_linear_reg_private(size=size, d=d, noise_std=noise_std, beta=beta, epsilon=epsilon)


# %%


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2,
    random_state=42,
)

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic 2D Data for SVM Classification")
plt.show()


X.shape
