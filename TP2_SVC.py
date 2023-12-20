import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from numpy import ravel


# Load data
csv_features = "~/Documents/App sup/app_sup/acsincome_ca_features.csv"
csv_labels = "~/Documents/App sup/app_sup/acsincome_ca_labels.csv"
features = pd.read_csv(csv_features)
labels = pd.read_csv(csv_labels)

# Separate features (X) and labels (y)
X_all = features 
y_all = labels 
#X_all, y_all = shuffle(X_all, y_all, random_state=0)


# Standardize the data
scaler = StandardScaler()


# Only use the first N samples to limit training time
num_samples = int(len(X_all) * 0.04)
X, y = X_all[:num_samples], y_all[:num_samples]

# Separate into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.8)


# Create a Random Forest model with specific hyperparameters
SVC_model = SVC( C= 10,
    kernel= 'rbf',
    gamma = 1)

#CORRELATION ENTRE CHAQU'UNES DES FEATURES ET LE LABEL

data_train = pd.concat([pd.DataFrame(X_train, columns=features.columns), y_train["PINCP"]], axis=1)
# Calculate the correlation matrix
corr_train = data_train.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_train, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix (FEATURES-LABELS)')
plt.show()

X_train = scaler.fit_transform(X_train)
SVC_model.fit(X_train, y_train["PINCP"])

X_test = scaler.transform(X_test)

predictions_rf = SVC_model.predict(X_test)





#CORRELATION ENTRE CHQU'UNES DES FEATURES ET La prediction
data_test = pd.concat([pd.DataFrame(X_test, columns=features.columns), pd.DataFrame(predictions_rf)], axis=1)
corr_test = data_test.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_test, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix (FEATURES-PRED)')
plt.show()



# Fit the permutation importance
result = permutation_importance(SVC_model, X_test, y_test["PINCP"], n_repeats=20, random_state=0)

# Get the importance scores and feature names
importance_scores = result.importances_mean
feature_names = X.columns

# Sort the features based on importance
sorted_idx = importance_scores.argsort()[::-1]

# Print the feature importance scores
print("Feature Importance Scores:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {importance_scores[i]}")

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance_scores)), importance_scores[sorted_idx], align="center")
plt.xticks(range(len(importance_scores)), feature_names[sorted_idx], rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Permutation Importance Score")
plt.title("Permutation Importance Scores")
plt.show()


