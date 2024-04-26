# %%
import numpy as np
import matplotlib.pyplot as plt
from generate_dp_data import generate_data
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from tqdm import tqdm
from utils import bin_private_data

# %%# Parameters
# Parameters
size = 300  # Number of samples
epsilon = 1  # Privacy parameter
d = 2  # Number of features
n_classes = 2  # Number of classes
cluster_std = [1, 1]
# Generate synthetic data
X, y = make_blobs(n_samples=size, centers=n_classes, n_features=d)
data = np.column_stack((X, y))

# Perturb data
private_data = generate_data(
    data, size, epsilon=epsilon, method="perturbed", shuffle=True
)
x_private, y_private = private_data[:, :d], bin_private_data(private_data[:, -1], 2)

# Fit SVM on private data
svm_private = SVC(kernel="linear")
svm_private.fit(x_private, np.round(y_private))

# Compute accuracies
accuracy_private = accuracy_score(np.round(y_private), svm_private.predict(x_private))
accuracy_original = accuracy_score(y, svm_private.predict(X))

# Plot decision boundaries
plt.figure(figsize=(12, 5))

# Plot decision boundary on original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50)
xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10)
)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = svm_private.decision_function(xy).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z,
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)
plt.scatter(
    svm_private.support_vectors_[:, 0],
    svm_private.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.title("Decision Boundary on Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot decision boundary on private data
plt.subplot(1, 2, 2)
plt.scatter(x_private[:, 0], x_private[:, 1], c=y_private, cmap=plt.cm.Paired, s=50)
xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10)
)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = svm_private.decision_function(xy).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z,
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)
plt.scatter(
    svm_private.support_vectors_[:, 0],
    svm_private.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.title("Decision Boundary on Private Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

print("Accuracy of SVM on private data:", accuracy_private)
print("Accuracy of SVM on original data:", accuracy_original)

# %%


def run_experiment_accuracy(
    N, methods, epsilon, num_trials=10, d=2, num_step=10, n_classes=2
):
    sizes = np.logspace(np.log10(100), np.log10(N), num=num_step, dtype=int)
    private_accuracies = {method: [] for method in methods}
    original_accuracies = {method: [] for method in methods}
    private_svm = SVC(kernel="linear")

    for method in methods:
        print(f"## COMPUTING method = {method} ##")
        for size in tqdm(sizes):
            private_accuracy_trials = []
            original_accuracy_trials = []
            for _ in range(num_trials):
                X_original, y_original = make_blobs(
                    n_samples=size, centers=n_classes, n_features=d, random_state=None
                )

                # Training on private data
                private_data = generate_data(
                    np.column_stack((X_original, y_original)),
                    size,
                    epsilon=epsilon,
                    method=method,
                    shuffle=True,
                )
                x_private, y_private = private_data[:, :d], bin_private_data(
                    private_data[:, -1], 2
                )
                y_private = np.ravel(y_private)

                private_svm.fit(x_private, np.round(y_private))
                accuracy_private = accuracy_score(
                    y_private, private_svm.predict(x_private)
                )
                private_accuracy_trials.append(accuracy_private)

                # Computing accuracy on non-private data
                accuracy_original = accuracy_score(
                    y_original, private_svm.predict(X_original)
                )
                original_accuracy_trials.append(accuracy_original)

            mean_private_accuracy = np.mean(private_accuracy_trials)
            std_private_accuracy = np.std(private_accuracy_trials) / np.sqrt(num_trials)
            private_accuracies[method].append(
                (mean_private_accuracy, std_private_accuracy)
            )

            mean_original_accuracy = np.mean(original_accuracy_trials)
            std_original_accuracy = np.std(original_accuracy_trials) / np.sqrt(
                num_trials
            )
            original_accuracies[method].append(
                (mean_original_accuracy, std_original_accuracy)
            )

    # Plotting for private data
    plt.figure(figsize=(10, 6))
    for method, accuracy_list in private_accuracies.items():
        mean_acc = [x[0] for x in accuracy_list]
        std_acc = [x[1] for x in accuracy_list]
        plt.errorbar(
            sizes,
            mean_acc,
            yerr=std_acc,
            label=f"{method}",
            fmt="-o",
            capsize=3,
        )
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy on Private Data vs Size of Dataset ($\\epsilon={epsilon}$)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting for original data using private SVM
    plt.figure(figsize=(10, 6))
    for method, accuracy_list in original_accuracies.items():
        mean_acc = [x[0] for x in accuracy_list]
        std_acc = [x[1] for x in accuracy_list]
        plt.errorbar(
            sizes,
            mean_acc,
            yerr=std_acc,
            label=f"{method}",
            fmt="-o",
            capsize=3,
        )
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy on Original Data Using Private SVM ($\\epsilon={epsilon}$)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Define parameters
N = 2000
epsilons = [0.1, 1, 10]
num_trials = 100
methods = [
    "super_regular_noise",
    "perturbed",
    "smooth_KS",
    "smooth_L2",
    "linear_stat_fit_grid",
]


d = 2
n_classes = 2
for epsilon in epsilons:
    # Run experiment
    run_experiment_accuracy(
        N, methods, epsilon=epsilon, num_trials=num_trials, d=d, n_classes=n_classes
    )


# %%
def run_experiment_accuracy_single_method(
    N, method, dimensions, epsilon, num_trials=10, num_step=10, n_classes=2
):
    sizes = np.logspace(np.log10(100), np.log10(N), num=num_step, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))

    private_svm = SVC(kernel="linear")

    private_accuracies_all = []
    original_accuracies_all = []

    for i, d in enumerate(dimensions):
        print(f"## COMPUTING method = {method}, dimension = {d} ##")
        private_accuracies = []
        original_accuracies = []
        for size in tqdm(sizes):
            private_accuracy_trials = []
            original_accuracy_trials = []
            for _ in range(num_trials):
                X_original, y_original = make_blobs(
                    n_samples=size, centers=n_classes, n_features=d, random_state=None
                )

                # Training on private data
                private_data = generate_data(
                    np.column_stack((X_original, y_original)),
                    size,
                    epsilon=epsilon,
                    method=method,
                    shuffle=True,
                )
                x_private, y_private = private_data[:, :d], bin_private_data(
                    private_data[:, -1], 2
                )
                y_private = np.ravel(y_private)

                private_svm.fit(x_private, np.round(y_private))
                accuracy_private = accuracy_score(
                    y_private, private_svm.predict(x_private)
                )
                private_accuracy_trials.append(accuracy_private)

                # Computing accuracy on non-private data
                accuracy_original = accuracy_score(
                    y_original, private_svm.predict(X_original)
                )
                original_accuracy_trials.append(accuracy_original)

            mean_private_accuracy = np.mean(private_accuracy_trials)
            std_private_accuracy = np.std(private_accuracy_trials) / np.sqrt(num_trials)
            private_accuracies.append((mean_private_accuracy, std_private_accuracy))

            mean_original_accuracy = np.mean(original_accuracy_trials)
            std_original_accuracy = np.std(original_accuracy_trials) / np.sqrt(
                num_trials
            )
            original_accuracies.append((mean_original_accuracy, std_original_accuracy))

        private_accuracies_all.append(private_accuracies)
        original_accuracies_all.append(original_accuracies)

    # Plotting for private data
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(dimensions):
        plt.errorbar(
            sizes,
            [x[0] for x in private_accuracies_all[i]],
            yerr=[x[1] for x in private_accuracies_all[i]],
            label=f"Dimension={d})",
            fmt="-o",
            capsize=3,
            color=colors[i],
        )
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy on Private Data vs Size of Dataset ($\\epsilon={epsilon}$)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting for original data using private SVM
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(dimensions):
        plt.errorbar(
            sizes,
            [x[0] for x in original_accuracies_all[i]],
            yerr=[x[1] for x in original_accuracies_all[i]],
            label=f"Dimension={d})",
            fmt="-o",
            capsize=3,
            color=colors[i],
        )
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy on Original Data Using Private SVM ($\\epsilon={epsilon}$)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Define parameters
N = 10000
epsilon = 10
num_trials = 10
method = "perturbed"
dimensions = [6, 7, 8]  # Add dimensions you want to test
n_classes = 2

# Run experiment
run_experiment_accuracy_single_method(
    N, method, dimensions, epsilon=epsilon, num_trials=num_trials, n_classes=n_classes
)
