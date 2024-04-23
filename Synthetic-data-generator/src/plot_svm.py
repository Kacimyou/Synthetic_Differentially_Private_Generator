# %%
import numpy as np
import matplotlib.pyplot as plt
from generate_dp_data import generate_data
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# %%

size = 5000
epsilon = 20
d = 2
n_classes = 2  # Set the number of classes

# Generate synthetic data
X, y = make_classification(
    n_samples=size,
    n_features=2,
    n_classes=n_classes,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2,
)
y = y.reshape((len(y), 1))
data = np.concatenate((X, y), axis=1)

private_data = generate_data(
    data, size, epsilon=epsilon, method="perturbated", shuffle=True
)
x_private = private_data[:, :d]
y_private = private_data[:, -1]
y_private_clipped = np.clip(y_private, 0, n_classes - 1)

# Round values to the nearest integer
y_private_rounded = np.round(y_private_clipped)

# Fit SVM on original data
svm_original = SVC(kernel="linear")
svm_original.fit(X, y.flatten())

# Fit SVM on private data
svm_private = SVC(kernel="linear")
svm_private.fit(x_private, y_private_rounded.flatten())

# Compute accuracy for both SVMs
accuracy_original = accuracy_score(y, svm_original.predict(X))
accuracy_private = accuracy_score(y_private_rounded, svm_private.predict(x_private))

# Plot the decision boundaries of both SVMs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Paired, s=50)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

num_points = 10
# Create grid to evaluate model using original dataset
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), num_points),
    np.linspace(X[:, 1].min(), X[:, 1].max(), num_points),
)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
print("Accuracy of SVM on original data:", accuracy_original)
print("Accuracy of SVM on private data:", accuracy_private)

# Calculate decision function values
Z = svm_original.decision_function(xy).reshape(xx.shape)

# Plot decision boundary and margins
ax.contour(
    xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)

ax.scatter(
    svm_original.support_vectors_[:, 0],
    svm_original.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.title("SVM on Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 2, 2)

plt.scatter(
    x_private[:, 0],
    x_private[:, 1],
    c=y_private_rounded.flatten(),
    cmap=plt.cm.Paired,
    s=50,
)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model using original dataset
xx, yy = np.meshgrid(
    np.linspace(x_private[:, 0].min(), x_private[:, 0].max(), num_points),
    np.linspace(x_private[:, 1].min(), x_private[:, 1].max(), num_points),
)
xy = np.vstack([xx.ravel(), yy.ravel()]).T

# Calculate decision function values
Z = svm_private.decision_function(xy).reshape(xx.shape)

# Plot decision boundary and margins
ax.contour(
    xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
ax.scatter(
    svm_private.support_vectors_[:, 0],
    svm_private.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.title("SVM on Private Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

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
