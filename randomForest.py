import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, train_size=.8)

X_train = scaler.fit_transform(X_train)
# Random Forest
# Hyperparameter tuning using Grid Search
param_grid_rf = {
    'n_estimators': [ 100, 200 , 300,500],
    'max_depth': [None,   30 ,40, 50,100],
    'min_samples_split': [ 10, 20 ,30, 40],
    'min_samples_leaf': [ 4 , 6, 8, 15]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid_rf, scoring='accuracy', cv=5)
grid_search_rf.fit(X_train, y_train["PINCP"])

best_model_rf = grid_search_rf.best_estimator_
best_model_param = grid_search_rf.best_params_
print("Best Model Parameters:")
print(best_model_param)


X_test = scaler.transform(X_test)
# Obtain predictions and train the model
predictions_rf = best_model_rf.predict(X_test)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.4f}')

# Display the classification report
report_rf = classification_report(y_test, predictions_rf)
print('Random Forest Classification Report:\n', report_rf)

# Display the confusion matrix
conf_matrix_rf = confusion_matrix(y_test, predictions_rf)
print('Random Forest Confusion Matrix:\n', conf_matrix_rf)

#Prédictions pour les autres états à partir du best model
print("Results on other state :\n")
test_features = pd.read_csv(csv_test)
test_res = pd.read_csv(csv_testRes)
X_test_norm = scaler.transform(test_features)
test_predictions = best_model_rf.predict(X_test_norm)

# Calculer l'accuracy
accuracy = accuracy_score(test_res, test_predictions)
print(f'Accuracy : {accuracy:.4f}')

# Afficher le classification report
report = classification_report(test_res, test_predictions)
print('Classification Report:\n', report)

# Afficher la confusion matrix
conf_matrix = confusion_matrix(test_res, test_predictions)
print('Confusion Matrix:\n', conf_matrix)