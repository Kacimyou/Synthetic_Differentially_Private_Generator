# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from generate_dp_data import generate_data
import scipy.stats
from tqdm import tqdm

N = 5000
epsilons = [10]
d = 1
beta = [1] * d
num_trials = 10
methods = [
    "super_regular_noise",
    "perturbed",
    "smooth_KS",
    "smooth_L2",
    "linear_stat_fit_grid",
    "linear_stat_fit_reg",
]

methods = ["perturbed"]


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
            color="tab:orange",
            capsize=3,  # Length of the error bar caps
            capthick=0.5,  # Thickness of the error bar caps
            barsabove=True,  # Error bars are drawn above the markers
        )

    plt.xlabel("Size of Database")
    plt.ylabel("Mean R2 Score")
    plt.grid()
    plt.title(f"Mean R2 Score vs Size of Database ($\\epsilon={epsilon}$)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


for epsilon in epsilons:
    run_experiment_R2(
        N, epsilon, methods, num_trials=num_trials, d=d, beta=beta, shuffle=False
    )


# %%
def run_experiment_beta(
    N, epsilon, methods, beta=[1], num_trials=10, d=1, noise_std=0.2, shuffle=True
):

    sizes = np.logspace(np.log10(100), np.log10(N), num=10, dtype=int)

    # colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    for i, method in enumerate(methods):
        print(f"## COMPUTING method = {method} ##")
        mse_scores = []
        stds = []
        num_trial = num_trials
        if method == "smooth_KS":
            num_trial = 250
        for size in tqdm(sizes):
            betas = []
            for j in tqdm(range(num_trial)):

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

            mean_beta = np.mean(np.array(betas))
            MSE_beta = (np.array(betas) - mean_beta) ** 2
            std = np.std(MSE_beta)
            stds.append(std)
            mse_scores.append(np.mean(MSE_beta))

        stds = np.array(stds) * (scipy.stats.norm.ppf(0.975) / np.sqrt(num_trials))
        # Plot mean MSE scores with variance for each method
        plt.errorbar(
            sizes,
            mse_scores,
            yerr=stds,
            label=f"Method: {method}",
            fmt="-o",
            linestyle="-",
            capsize=3,  # Length of the error bar caps
            capthick=0.5,  # Thickness of the error bar caps
            barsabove=True,  # Error bars are drawn above the markers
        )

    plt.xlabel("Size of Initial Database")
    plt.ylabel("Mean Squared Error (MSE) of Coefficient Estimates ")
    plt.yscale("log")
    plt.grid()
    plt.title(
        f"MSE of Coefficient Estimates vs Size of Database ($\\epsilon={epsilon}$)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


# methods = ["smooth", "perturbed"]
for epsilon in epsilons:

    run_experiment_beta(
        N, epsilon, methods, num_trials=num_trials, d=d, beta=beta, shuffle=False
    )


# %%


def plot_linear_reg_private(size, d, noise_std, beta, epsilon, method):
    X = np.random.randn(size, d)
    y = np.dot(X, beta).reshape(size, 1) + np.random.randn(size, 1) * noise_std
    data = np.concatenate((X, y), axis=1)

    # Assuming you have implemented the generate_smooth_data function correctly
    private_data = generate_data(
        data, size, epsilon, method=method, shuffle=True, rescaling=True
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
    plt.title("Linear Regression on DP Synthetic Data and its Non Private input")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Generate synthetic data
size = 300
d = 1
noise_std = 1
beta = 5
epsilon = 2
method = "perturbed"

plot_linear_reg_private(
    size=size, d=d, noise_std=noise_std, beta=beta, epsilon=epsilon, method=method
)

# %%


def run_experiment_R2_single_method(
    N,
    epsilon,
    method,
    dimensions,
    beta=[1],
    num_trials=10,
    num_step=10,
    noise_std=0.2,
    shuffle=True,
):

    sizes = np.logspace(np.log10(100), np.log10(N), num=num_step, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))

    for i, d in enumerate(dimensions):
        print(f"## COMPUTING method = {method}, dimension = {d} ##")
        mean_r2_scores = []
        stds = []
        beta_d = beta * d
        print(beta_d)
        for size in tqdm(sizes):
            r2_scores = []
            for _ in tqdm(range(num_trials)):

                X = np.random.randn(size, d)
                y = (
                    np.dot(X, beta_d).reshape(size, 1)
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
        # Plot mean R2 scores with variance for each dimension
        plt.errorbar(
            sizes,
            mean_r2_scores,
            yerr=stds,
            label=f"Dimension = {d}",
            fmt="-o",
            linestyle="-",
            color=colors[i],
            capsize=3,  # Length of the error bar caps
            capthick=0.5,  # Thickness of the error bar caps
            barsabove=True,  # Error bars are drawn above the markers
        )

    plt.xlabel("Size of Database")
    plt.ylabel("Mean R2 Score")
    plt.grid()
    plt.ylim(0, 1)  # Limit y-axis between 0 and 1
    plt.title(f"Mean R2 Score vs Size of Database ($\\epsilon={epsilon}$)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


# Example usage:
dimensions = [1, 2, 3, 4]
run_experiment_R2_single_method(
    N,
    epsilon=10,
    method="perturbed",
    dimensions=dimensions,
    num_trials=num_trials,
    beta=[1],  # Use max dimension for beta
    shuffle=False,
)
