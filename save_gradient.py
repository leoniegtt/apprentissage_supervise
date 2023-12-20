import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
csv_features = "/home/goutte/5A/app_sup/acsincome_ca_features.csv"
csv_labels = "/home/goutte/5A/app_sup/acsincome_ca_labels.csv"
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

print('hanam3ak')
# Create a Random Forest model with specific hyperparameters
gb_model = GradientBoostingClassifier( learning_rate= 0.1, max_depth= 5, n_estimators= 50)

X_train = scaler.fit_transform(X_train)
gb_model.fit(X_train, y_train["PINCP"])

X_test = scaler.transform(X_test)

predictions_gb = gb_model.predict(X_test)


#CORRELATION ENTRE CHQU'UNES DES FEATURES ET LE LABEL
()
data_train = pd.concat([pd.DataFrame(X_train, columns=features.columns), y_train["PINCP"]], axis=1)
# Calculate the correlation matrix
corr_train = data_train.corr(numeric_only=False)
print(corr_train)

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_train, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


#CORRELATION ENTRE CHQU'UNES DES FEATURES ET La prediction
data_test = pd.concat([pd.DataFrame(X_test, columns=features.columns), pd.DataFrame(predictions_gb)], axis=1)
corr_test = data_train.corr(numeric_only=False)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_test, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
