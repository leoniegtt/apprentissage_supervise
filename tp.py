import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

csv_features = "~/Documents/App sup/app_sup/acsincome_ca_features.csv"
csv_labels = "~/Documents/App sup/app_sup/acsincome_ca_labels.csv"

csv_test = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_co_allfeaturesTP2.csv"
csv_testRes = "~/Documents/App sup/app_sup/TP2-complementaryData/acsincome_co_labelTP2.csv"

features = pd.read_csv(csv_features)
labels = pd.read_csv(csv_labels)


# Séparer les caractéristiques (X) et les étiquettes (y)
X_all = features 
y_all = labels 
X_all, y_all = shuffle(X_all, y_all, random_state=0)

#standardisation des données
scaler = StandardScaler()

# only use the first N samples to limit training time
num_samples = int(len(X_all)*0.01)
X, y = X_all[:num_samples], y_all[:num_samples]

#separate into training set et testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .8)
X_train = scaler.fit_transform(X_train)

#SVM
#1-validation croisée
# Créer une instance du modèle SVM
svm_model = SVC() 

#scores = cross_val_score(svm_model, X_train, y_train, cv=nb_plis, scoring='accuracy')

#récup modèle
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': [ 'rbf', 'poly'],
    'gamma': [0.1]
}
nb_plis=5
kf = KFold(n_splits=nb_plis, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring='accuracy', cv=kf)
grid_search.fit(X_train, y_train["PINCP"])

best_model = grid_search.best_estimator_
print(grid_search.best_params_)

X_test = scaler.transform(X_test)
#obtenir prédictions et entrainer modèle
predictions = best_model.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy : {accuracy:.4f}')

# Afficher le classification report
report = classification_report(y_test, predictions)
print('Classification Report y_test et predictions en svm:\n', report)

# Afficher la confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print('Confusion Matrix y_test et predictions en svm :\n', conf_matrix)

#Prédictions pour les autres états à partir du best model
test_features = pd.read_csv(csv_test)
test_res = pd.read_csv(csv_testRes)
X_test_norm = scaler.transform(test_features)
test_predictions = best_model.predict(X_test_norm)

# Calculer l'accuracy
accuracy = accuracy_score(test_res, test_predictions)
print(f'Accuracy : {accuracy:.4f}\n')

# Afficher le classification report
report = classification_report(test_res, test_predictions)
print('Classification Report:\n test res et test prediction:\n', report)

# Afficher la confusion matrix
conf_matrix = confusion_matrix(test_res, test_predictions)
print('Confusion Matrix test res et test prediction:\n', conf_matrix)

