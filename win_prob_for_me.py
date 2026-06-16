import joblib
import numpy as np
grid = joblib.load("cricket_win_prob_model.pkl")
input = np.array([[15, 123, 8, 37, 5, 132, 5, 2, 123/15, 132/5]])
pred_prob = grid.predict_proba(input)[:, 1]
print(pred_prob)