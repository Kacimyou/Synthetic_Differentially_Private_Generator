# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from generate_dp_data import generate_data


def plot_linear_reg_private(size, d, noise_std, beta, epsilon, method):
    X = np.random.randn(size, d)
    y = np.dot(X, beta).reshape(size, 1) + np.random.randn(size, 1) * noise_std
    data = np.concatenate((X, y), axis=1)

    # Assuming you have implemented the generate_smooth_data function correctly
    private_data = generate_data(
        data, size, epsilon, method=method, shuffle=True, rescaling=True, verbose=1
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
    plt.grid()
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
    plt.title("Linear Regression on Smooth histogramm generated data")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Generate synthetic data
size = 500
d = 1
noise_std = 2
beta = 5
epsilon = 1


method = "perturbed"

plot_linear_reg_private(
    size=size, d=d, noise_std=noise_std, beta=beta, epsilon=epsilon, method=method
)
