import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from numpy import ravel

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# Load data
csv_features = "~/Documents/App sup/app_sup/acsincome_ca_features.csv"
csv_labels = "~/Documents/App sup/app_sup/acsincome_ca_labels.csv"
csv_groupSex = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_ca_group_Sex.csv"
csv_groupRace = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_ca_group_Race.csv"
features = pd.read_csv(csv_features)
labels = pd.read_csv(csv_labels)
maskSex = pd.read_csv(csv_groupSex)
maskRace = pd.read_csv(csv_groupRace)

# Separate features (X) and labels (y)
X_all = features 
y_all = labels 
#X_all, y_all = shuffle(X_all, y_all, random_state=0)


# Standardize the dataw
scaler = StandardScaler()


# Only use the first N samples to limit training time
num_samples = int(len(X_all) * 0.04)
X, y = X_all[:num_samples], y_all[:num_samples]

# Separate into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.8)
X_train = scaler.fit_transform(X_train)

models = [
    ("SVM", SVC(C = 10.0, kernel = 'rbf')),
    ("Adaboost", AdaBoostClassifier(learning_rate = 1, n_estimators = 200)),
    ("GradientBoosting", GradientBoostingClassifier(learning_rate=0.1, n_estimators=200)),
    ("RandomForest", RandomForestClassifier(max_depth=10, n_estimators=  200))
]

sex_column_index =  8
feature_values = np.unique(X_test['RAC1P'])
feature = 'RAC1P'
print(feature_values)
for name , model in models : 
    model_instance = model
    model_instance.fit(X_train, y_train["PINCP"])
    # Iterate over each unique value of 'SEX' and compute confusion matrix
    for feature_value in feature_values:
        # Find the indices where 'SEX' is equal to the current value
        subset_indices = np.where(X_test[feature].values == feature_value)[0]

        # Use these indices to extract the subset of X_test and y_test
        X_test_subset = X_test.iloc[subset_indices]  # Use .iloc to index rows
        y_test_subset = y_test.iloc[subset_indices]

        # Assuming PINCP is a column in y_test, you can extract it like this
        PINCP_subset = y_test_subset["PINCP"]

        X_test_subset = scaler.transform(X_test_subset)
        
        # Make predictions for the subset
        predictions_subset = model_instance.predict(X_test_subset)

        # Compute and print confusion matrix for the subset
        confusion_subset = confusion_matrix(PINCP_subset, predictions_subset)
        print(f"Confusion Matrix for model { name } for the feature {feature} = {feature_value}:\n{confusion_subset}")

        # Optionally, calculate and print accuracy, classification report, etc.
        accuracy_subset = accuracy_score(PINCP_subset, predictions_subset)
        print(f"Accuracy for the model {name } for the feature {feature} = {feature_value}: {accuracy_subset}\n")