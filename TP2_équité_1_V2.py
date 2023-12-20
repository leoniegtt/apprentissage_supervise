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
num_samples = int(len(X_all) * 0.1)
X, y = X_all[:num_samples], y_all[:num_samples]

# Separate into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45, train_size=.8)


models = [
    ("SVM", SVC(C = 10.0, kernel = 'rbf')),
    ("Adaboost", AdaBoostClassifier(learning_rate = 1, n_estimators = 200)),
    ("GradientBoosting", GradientBoostingClassifier(learning_rate=0.1, n_estimators=200)),
    ("RandomForest", RandomForestClassifier(max_depth=10, n_estimators=  200))
]

feature = 'RAC1P'

for name, model in models:
    feature_values = np.unique(X_train[feature])
    for feature_value in feature_values:
        subset_indices = np.where(X_train[feature].values == feature_value)[0]
        X_train_subset = X_train.iloc[subset_indices]
        
        y_train_subset = y_train.iloc[subset_indices]
        PINCP_subset = y_train_subset["PINCP"]
        
        # Standardize the subset while preserving column names
        scaler.fit(X_train_subset)
        X_train_subset = pd.DataFrame(scaler.transform(X_train_subset), columns=X_train.columns)
        
        model.fit(X_train_subset, PINCP_subset)
        
        # Standardize the test set with the same scaler
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        predictions_subset = model.predict(X_test_scaled)
        
        # Compute and print confusion matrix for the subset
        confusion_subset = confusion_matrix(y_test["PINCP"], predictions_subset)
        print(confusion_subset)
        
        # Extract TP, FP, TN, FN from confusion matrix
        TP = confusion_subset[1, 1]
        FP = confusion_subset[0, 1]
        TN = confusion_subset[0, 0]
        FN = confusion_subset[1, 0]
        
        # Calculate metrics
        accuracy_subset = (TP + TN) / (TP + TN + FP + FN)
        recall_subset = TP / (TP + FN)
        false_positive_rate_subset = FP / (FP + TN)
        
        # Print metrics
        print(f"Metrics for model {name} for the feature {feature} = {feature_value}:")
        print(f"Accuracy: {accuracy_subset}")
        print(f"Recall: {recall_subset}")
        print(f"False Positive Rate: {false_positive_rate_subset}\n")