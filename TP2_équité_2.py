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
csv_noSex = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_ca_features_without_sex.csv"
csv_noRace = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_ca_features_without_sex.csv"
features = pd.read_csv(csv_features)
labels = pd.read_csv(csv_labels)
noSexe = pd.read_csv(csv_noSex)
noRace = pd.read_csv(csv_noRace)

# Separate features (X) and labels (y)
X_all = noSexe

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

for name, model in models:
    model.fit(X_train, y_train["PINCP"])
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy for {name}: {accuracy:.4f}')

    # Afficher le classification report avec zero_division parameter
    report = classification_report(y_test, predictions, zero_division=1)
    print(f"Classification Report for {name}:\n", report)

    # Afficher la confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix for {name}:\n", conf_matrix)
    # Extract TP, FP, TN, FN from confusion matrix
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    
    # Calculate metrics
    accuracy_subset = (TP + TN) / (TP + TN + FP + FN)
    recall_subset = TP / (TP + FN)
    false_positive_rate_subset = FP / (FP + TN)
    
    # Print metrics
    print(f"Metrics for model {name} ")
    print(f"Accuracy: {accuracy_subset}")
    print(f"Recall: {recall_subset}")
    print(f"False Positive Rate: {false_positive_rate_subset}\n")