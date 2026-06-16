import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit



X_data = np.load('X_data.npy')[:, :-1]
y_data = np.load('y_data.npy')
weight_data = np.load('weight_data.npy')
match_ids = np.load('match_id_data.npy')

gss = GroupShuffleSplit(test_size=0.15, random_state=71)

train_idx, test_idx = next(
    gss.split(X_data, y_data, groups=match_ids)
)

X_train, X_test = X_data[train_idx], X_data[test_idx]
y_train, y_test = y_data[train_idx], y_data[test_idx]
weight_train, weight_test = weight_data[train_idx], weight_data[test_idx]


model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, verbose=1))
])

param_grid = {
    "logreg__C": [1, 10, 20, 50],  # regularization strength
    "logreg__l1_ratio": [0]
}

gkf = GroupKFold(n_splits=5)

grid = GridSearchCV(model, param_grid, cv=gkf.split(X_train, y_train, groups=match_ids[train_idx]), verbose=2)  # cv=5 means 5-fold cross-validation

# Fit to training data
grid.fit(X_train, y_train, logreg__sample_weight=weight_train)

# Best hyperparameters
print(grid.best_params_)

# Best score
print(grid.best_score_)
print("Iterations to converge:", grid.best_estimator_.named_steps['logreg'].n_iter_)
joblib.dump(grid.best_estimator_, "cricket_win_prob_model.pkl")
