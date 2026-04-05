import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# PART 1: LOADING DATA FROM CSV
# ==========================================
csv_filename = 'Dataset/flattened_dataset/flattened_breadboards.csv'

print(f"Step 1: Loading flattened image data from {csv_filename}...")
try:
    df = pd.read_csv(csv_filename)
    print(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")
except FileNotFoundError:
    print(f"Error: Could not find '{csv_filename}'.")
    print("Please run 'image_flattener.py' first to generate the dataset!")
    exit()

# ==========================================
# PART 2: TRADITIONAL MACHINE LEARNING
# ==========================================
print("Step 2: Training Traditional ML Models on the CSV Data...\n")

# 1. Split data into Features (X) and Labels (y)
X = df.drop('label', axis=1) # Everything except the label column
y = df['label']              # Only the label column

# 2. Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL 1: Support Vector Machine (SVM) ---
print("Training Support Vector Machine (SVM)...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"--> SVM Accuracy: {svm_accuracy * 100:.2f}%")


# --- MODEL 2: Random Forest (Ensemble Learning) ---
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"--> Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# --- MODEL 3: Logistic Regression ---
print("\nTraining Logistic Regression...")
# max_iter is set high because image data takes longer to converge
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"--> Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# --- MODEL 4: K-Nearest Neighbors (KNN) ---
print("\nTraining K-Nearest Neighbors (KNN)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"--> KNN Accuracy: {knn_accuracy * 100:.2f}%")

# --- MODEL 5: Gradient Boosting ---
print("\nTraining Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f"--> Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")

# --- MODEL 6: Multi-Layer Perceptron (Basic Neural Network) ---
print("\nTraining Multi-Layer Perceptron (MLP)...")
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

mlp_predictions = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f"--> MLP Accuracy: {mlp_accuracy * 100:.2f}%\n")

# --- DETAILED REPORT FOR THE WINNING TRADITIONAL MODEL ---
# 1. Group all the results together into a dictionary
model_results = {
    'Support Vector Machine (SVM)': {'accuracy': svm_accuracy, 'predictions': svm_predictions, 'model': svm_model},
    'Random Forest': {'accuracy': rf_accuracy, 'predictions': rf_predictions, 'model': rf_model},
    'Logistic Regression': {'accuracy': lr_accuracy, 'predictions': lr_predictions, 'model': lr_model},
    'K-Nearest Neighbors (KNN)': {'accuracy': knn_accuracy, 'predictions': knn_predictions, 'model': knn_model},
    'Gradient Boosting': {'accuracy': gb_accuracy, 'predictions': gb_predictions, 'model': gb_model},
    'Multi-Layer Perceptron (MLP)': {'accuracy': mlp_accuracy, 'predictions': mlp_predictions, 'model': mlp_model}
}

# 2. Automatically find the name of the baseline model with the highest accuracy
best_model_name = max(model_results, key=lambda k: model_results[k]['accuracy'])
print(f"\n--- Baseline Winner Identified: {best_model_name} ({model_results[best_model_name]['accuracy']*100:.2f}%) ---")

# ==========================================
# PART 3: HYPERPARAMETER TUNING (RandomizedSearchCV)
# ==========================================
print(f"\nStep 3: Tuning hyperparameters for the winning model ({best_model_name})...")

# Define hyperparameter search spaces for all models. 
# The script will dynamically select the one that matches our winner.
tuning_grids = {
    'Support Vector Machine (SVM)': {
        'estimator': SVC(random_state=42),
        'params': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    },
    'Random Forest': {
        'estimator': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    },
    'Logistic Regression': {
        'estimator': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10, 100]}
    },
    'K-Nearest Neighbors (KNN)': {
        'estimator': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
    },
    'Gradient Boosting': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    },
    'Multi-Layer Perceptron (MLP)': {
        'estimator': MLPClassifier(max_iter=500, random_state=42),
        'params': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]}
    }
}

# Retrieve the specific estimator and parameter grid for the winning model
tuned_estimator = tuning_grids[best_model_name]['estimator']
param_distributions = tuning_grids[best_model_name]['params']

# Initialize RandomizedSearchCV
# n_iter=10 limits the search to 10 random combinations to keep training time reasonable.
# n_jobs=-1 tells scikit-learn to use ALL of your computer's CPU cores to speed things up!
random_search = RandomizedSearchCV(
    estimator=tuned_estimator, 
    param_distributions=param_distributions, 
    n_iter=10, 
    cv=3, 
    scoring='accuracy', 
    random_state=42, 
    n_jobs=-1, 
    verbose=1
)

# Run the search
random_search.fit(X_train, y_train)

# Grab the newly optimized model
best_tuned_model = random_search.best_estimator_
best_tuned_predictions = best_tuned_model.predict(X_test)
best_tuned_accuracy = accuracy_score(y_test, best_tuned_predictions)

print("\n=========================================================")
print(f" FINAL REPORT: Optimized {best_model_name} ")
print("=========================================================")
print(f"Best Hyperparameters Found:\n{random_search.best_params_}\n")
print(f"Optimized Accuracy: {best_tuned_accuracy * 100:.2f}%\n")
print(classification_report(y_test, best_tuned_predictions))

# 3. Export the tuned winning model
print(f"Exporting the optimized model...")
joblib.dump(best_tuned_model, 'Models/best_traditional_model.joblib')
print("Model saved successfully as 'Models/best_traditional_model.joblib'!")