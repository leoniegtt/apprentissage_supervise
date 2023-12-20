import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
csv_features = "~/Documents/App sup/app_sup/acsincome_ca_features.csv"
csv_labels = "~/Documents/App sup/app_sup/acsincome_ca_labels.csv"

csv_test = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_ne_allfeaturesTP2.csv"
csv_testRes = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_ne_labelTP2.csv"
features = pd.read_csv(csv_features)
labels = pd.read_csv(csv_labels)

# Separate features (X) and labels (y)
X_all = features 
y_all = labels 
X_all, y_all = shuffle(X_all, y_all, random_state=0)

# Standardize the data
scaler = StandardScaler()


# Only use the first N samples to limit training time
num_samples = int(len(X_all) * 0.01)
X, y = X_all[:num_samples], y_all[:num_samples]

# Separate into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.8)
X_train = scaler.fit_transform(X_train)
# AdaBoost
# Hyperparameter tuning using Grid Search
param_grid_ada = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1]
}

grid_search_ada = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid_ada, scoring='accuracy', cv=5)
grid_search_ada.fit(X_train, y_train["PINCP"])

best_model_ada = grid_search_ada.best_estimator_
print(grid_search_ada.best_params_)
# Obtain predictions and train the model
X_test = scaler.transform(X_test)
predictions_ada = best_model_ada.predict(X_test)

# Evaluate AdaBoost model
accuracy_ada = accuracy_score(y_test, predictions_ada)
print(f'AdaBoost Accuracy: {accuracy_ada:.4f}')

# Display the classification report
report_ada = classification_report(y_test, predictions_ada)
print('AdaBoost Classification Report:\n', report_ada)

# Display the confusion matrix
conf_matrix_ada = confusion_matrix(y_test, predictions_ada)
print('AdaBoost Confusion Matrix:\n', conf_matrix_ada)

#Prédictions pour les autres états à partir du best model
test_features = pd.read_csv(csv_test)
test_res = pd.read_csv(csv_testRes)
X_test_norm = scaler.transform(test_features)
test_predictions = best_model_ada.predict(X_test_norm)

# Calculer l'accuracy
accuracy = accuracy_score(test_res, test_predictions)
print(f'Accuracy : {accuracy:.4f}')

# Afficher le classification report
report = classification_report(test_res, test_predictions)
print('Classification Report:\n test res et test prediction :\n', report)
# Afficher la confusion matrix
conf_matrix = confusion_matrix(test_res, test_predictions)
print('Confusion Matrix test res et test prediction:\n', conf_matrix)
