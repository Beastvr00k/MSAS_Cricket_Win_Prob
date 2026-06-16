import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

X_data = np.load('X_data.npy')[:, 5:]
y_data = np.load('y_data.npy')
mask = y_data.astype(bool)
X_data = X_data[mask]
weight_data = np.load('weight_data.npy')
weight_data = weight_data[mask]
match_id_data = np.load('match_id_data.npy')
match_id_data = match_id_data[mask]
new_y_data = X_data[:, -1]
X_data = np.delete(X_data, -1, 1)
pressure = (X_data[:, -1] - X_data[:, -2]).reshape(-1, 1)
X_data = np.hstack((X_data, pressure))

gss = GroupShuffleSplit(test_size=0.15, random_state=71)

train_idx, test_idx = next(
    gss.split(X_data, new_y_data, groups=match_id_data)
)

X_train, X_test = X_data[train_idx], X_data[test_idx]
y_train, y_test = new_y_data[train_idx], new_y_data[test_idx]
weight_train, weight_test = weight_data[train_idx], weight_data[test_idx]

model = Pipeline([
    ("rf", RandomForestRegressor(random_state=42, verbose=1, n_jobs=-1))
])

param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5]
}

gkf = GroupKFold(n_splits=5)
grid = GridSearchCV(model, param_grid, cv=gkf.split(X_train, y_train, groups=match_id_data[train_idx]), verbose=2, error_score="raise")  # cv=5 means 5-fold cross-validation

# Fit to training data
grid.fit(X_train, y_train, rf__sample_weight=weight_train)

# Best hyperparameters
print(grid.best_params_)

# Best score
print(grid.best_score_)
print(grid.best_estimator_)
joblib.dump(grid.best_estimator_, "chase_model.pkl")

