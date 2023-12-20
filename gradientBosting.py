import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
csv_features = "~/Documents/App sup/app_sup/acsincome_ca_features.csv"
csv_labels = "~/Documents/App sup/app_sup/acsincome_ca_labels.csv"
features = pd.read_csv(csv_features)
labels = pd.read_csv(csv_labels)

csv_test = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_co_allfeaturesTP2.csv"
csv_testRes = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_co_labelTP2.csv"

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
# Gradient Boosting
# Hyperparameter tuning using Grid Search
param_grid_gb = {
    'n_estimators': [50, 100, 150,200,500],
    'learning_rate': [ 0.1, 1,2,3],
    'max_depth': [3, 5, 7,10]
}

grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid_gb, scoring='accuracy', cv=5)
grid_search_gb.fit(X_train, y_train["PINCP"])

best_model_gb = grid_search_gb.best_estimator_
print(grid_search_gb.best_params_)
# Obtain predictions and train the model
X_test = scaler.fit_transform(X_test)
predictions_gb = best_model_gb.predict(X_test)

# Evaluate Gradient Boosting model
accuracy_gb = accuracy_score(y_test, predictions_gb)
print(f'Gradient Boosting Accuracy: {accuracy_gb:.4f}')

# Display the classification report
report_gb = classification_report(y_test, predictions_gb)
print('Gradient Boosting Classification Report:\n', report_gb)

# Display the confusion matrix
conf_matrix_gb = confusion_matrix(y_test, predictions_gb)
print('Gradient Boosting Confusion Matrix:\n', conf_matrix_gb)


#Prédictions pour les autres états à partir du best model
print('test on new  dataset:\n')
test_features = pd.read_csv(csv_test)
test_res = pd.read_csv(csv_testRes)
X_test_norm = scaler.transform(test_features)
test_predictions = best_model_gb.predict(X_test_norm)

# Calculer l'accuracy
accuracy = accuracy_score(test_res, test_predictions)
print(f'Accuracy : {accuracy:.4f}')

# Afficher le classification report
report = classification_report(test_res, test_predictions)
print('Classification Report:\n', report)

# Afficher la confusion matrix
conf_matrix = confusion_matrix(test_res, test_predictions)
print('Confusion Matrix:\n', conf_matrix)
