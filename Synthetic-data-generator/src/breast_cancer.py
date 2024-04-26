# %%
import numpy as np
import matplotlib.pyplot as plt
from generate_dp_data import generate_data
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from utils import bin_private_data


#### IMPORT DATA #####
data = load_iris()
# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# Add species column
df["target"] = data.target


# %%
iris = load_iris()

# Convert the Iris dataset to a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

# %%
# Number of bootstrap samples to generate
num_bootstrap_samples = 50

# Bootstrap the dataset
bootstrap_samples = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = iris_df.sample(n=len(iris_df), replace=True)
    bootstrap_samples.append(bootstrap_sample)

# Concatenate the bootstrap samples into a single DataFrame
bootstrap_df = pd.concat(bootstrap_samples, ignore_index=True)

# Shuffle the rows to mix the original and bootstrap samples
bootstrap_df = bootstrap_df.sample(frac=1, random_state=42).reset_index(drop=True)
df = bootstrap_df

df.head

# %%
## NON PRIVATE SCHEME #####
# Split data into features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define XGBoost model
model = xgb.XGBClassifier()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Concatenate features and target into a single DataFrame
df_train = X_train.copy()
df_train["target"] = y_train

# Create pairplot
sns.pairplot(df_train, hue="target", palette="Set1")
plt.suptitle("Non-Private Scheme: Pairplot of Training Data", y=1.02)
plt.show()
# %%
## PRIVATE SCHEME #####
epsilon = 1000
# Convert DataFrame to numpy array for privacy scheme
data = np.concatenate((X.values, y.values.reshape(-1, 1)), axis=1)

# print(data)
# Generate private version of the training data
private_data = generate_data(
    data,
    size=len(data),
    epsilon=epsilon,
    method="super_regular_noise",
    shuffle=False,
    verbose=1,
)

# Remap the private version of the target variable to DataFrame format
private_data_df = pd.DataFrame(private_data, columns=df.columns)

# Post-processing step: Map the target column to its closest integer


# private_data_df["target"] = bin_private_data(private_data_df["target"], 3)

# Split the private training and test data into features (X) and target (y)


X_private = private_data_df.drop("target", axis=1)
y_private = private_data_df["target"]

print(y_private.unique(), np.unique(bin_private_data(y_private, 3)))


# X_private_train, X_private_test, y_private_train, y_private_test = train_test_split(
#     X_private, y_private, test_size=0.2, random_state=42
# )

# # Define XGBoost model for private scheme
# model_private = xgb.XGBClassifier()

# # Fit the model with private training data
# model_private.fit(X_private_train, y_private_train)

# # Make predictions on private test data
# y_private_pred = model_private.predict(X_private_test)

# # Evaluate accuracy of the private model
# accuracy_private = accuracy_score(y_private_test, y_private_pred)
# print("Accuracy (Private):", accuracy_private)


# Create pairplot
sns.pairplot(private_data_df, hue="target", palette="Set1")
plt.suptitle("Non-Private Scheme: Pairplot of Training Data", y=1.02)
plt.show()

# %%
