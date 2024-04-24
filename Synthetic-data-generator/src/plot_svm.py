# %%
import numpy as np
import matplotlib.pyplot as plt
from generate_dp_data import generate_data
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from tqdm import tqdm


# %%
size = 4000
epsilon = 0.5
d = 2
n_classes = 2

# Generate synthetic data
X, y = make_blobs(n_samples=size, centers=n_classes, n_features=d)
data = np.column_stack((X, y))

# Perturb data
private_data = generate_data(
    data, size, epsilon=epsilon, method="perturbed", shuffle=True
)
x_private, y_private = private_data[:, :d], np.clip(
    private_data[:, -1], 0, n_classes - 1
)

# Fit SVMs
svm_original, svm_private = SVC(kernel="linear"), SVC(kernel="linear")
svm_original.fit(X, y)
svm_private.fit(x_private, np.round(y_private))

# Compute accuracies
accuracy_original = accuracy_score(y, svm_original.predict(X))
accuracy_private = accuracy_score(np.round(y_private), svm_private.predict(x_private))

# Plot decision boundaries
plt.figure(figsize=(12, 5))
for i, svm, title in zip(
    [1, 2], [svm_original, svm_private], ["SVM on Original Data", "SVM on Private Data"]
):
    plt.subplot(1, 2, i)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50)
    xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10)
    )
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = svm.decision_function(xy).reshape(xx.shape)
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
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

print("Accuracy of SVM on original data:", accuracy_original)
print("Accuracy of SVM on private data:", accuracy_private)

# %%

# Define parameters
size = 120
epsilon = 2
d = 2
n_classes = 3  # Set the number of classes

# Generate synthetic data
X, y = make_classification(
    n_samples=size,
    n_features=2,
    n_classes=n_classes,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=3,
)
y = y.reshape((len(y), 1))
data = np.concatenate((X, y), axis=1)

private_data = generate_data(
    data, size, epsilon=epsilon, method="perturbated", shuffle=True, verbose=True
)
x_private = private_data[:, :d]
y_private = private_data[:, -1]

# Calculate bin edges
bin_edges = np.linspace(min(y_private), max(y_private), n_classes + 1)

# Assign each y_private value to a bin interval
y_private_binned = (
    np.digitize(y_private, bin_edges) - 1
)  # Subtract 1 to start from 0 index
y_private_binned = y_private_binned.reshape((len(y), 1))

# Fit SVM on original data
svm_original = SVC(kernel="linear")
svm_original.fit(X, y.flatten())

# Fit SVM on private data
svm_private = SVC(kernel="linear")
svm_private.fit(x_private, y_private_binned.flatten())

# Compute accuracy for both SVMs
accuracy_original = accuracy_score(y, svm_original.predict(X))
accuracy_private = accuracy_score(y_private_binned, svm_private.predict(x_private))

print("Accuracy of SVM on original data:", accuracy_original)
print("Accuracy of SVM on private data:", accuracy_private)

# Determine color limits
color_min = min(y.flatten())
color_max = max(y.flatten())

# Plotting without color bar
plt.figure(figsize=(12, 6))

# Scatter plot of original data
plt.subplot(1, 2, 1)
plt.scatter(
    X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Paired, vmin=color_min, vmax=color_max
)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Scatter plot of private data
plt.subplot(1, 2, 2)
plt.scatter(
    x_private[:, 0],
    x_private[:, 1],
    c=y_private_binned,
    cmap=plt.cm.Paired,
    vmin=color_min,
    vmax=color_max,
)
plt.title("Private Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

# %%


def run_experiment_accuracy(
    N, methods, epsilon, num_trials=10, d=2, num_step=10, n_classes=2
):
    sizes = np.logspace(np.log10(100), np.log10(N), num=num_step, dtype=int)
    accuracies = {method: [] for method in methods}

    for method in methods:
        print(f"## COMPUTING method = {method} ##")
        for size in tqdm(sizes):
            accuracy_trials = []
            for _ in range(num_trials):
                X, y = make_blobs(n_samples=size, centers=n_classes, n_features=d)
                private_data = generate_data(
                    np.column_stack((X, y)),
                    size,
                    epsilon=epsilon,
                    method=method,
                    shuffle=True,
                )
                x_private, y_private = private_data[:, :d], np.clip(
                    private_data[:, -1], 0, n_classes - 1
                )

                svm_private = SVC(kernel="linear")
                svm_private.fit(x_private, np.round(y_private))
                accuracy_private = accuracy_score(y, svm_private.predict(X))

                accuracy_trials.append(accuracy_private)

            mean_accuracy = np.mean(accuracy_trials)
            std_accuracy = np.std(accuracy_trials)
            accuracies[method].append((mean_accuracy, std_accuracy))

    plt.figure(figsize=(10, 6))
    for method, accuracy_list in accuracies.items():
        mean_acc = [x[0] for x in accuracy_list]
        std_acc = [x[1] for x in accuracy_list]
        plt.errorbar(
            sizes,
            mean_acc,
            yerr=std_acc,
            label=f"Method = {method}",
            fmt="-o",
            capsize=3,
        )
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Size of Dataset ($\\epsilon={epsilon}$)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Define parameters
N = 5000
epsilon = 1
num_trials = 10
methods = ["smooth", "perturbed"]
d = 2
n_classes = 2

# Run experiment
run_experiment_accuracy(
    N, methods, epsilon=epsilon, num_trials=num_trials, d=d, n_classes=n_classes
)
