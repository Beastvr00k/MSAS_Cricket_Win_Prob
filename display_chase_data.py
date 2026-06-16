import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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

grid = joblib.load("chase_model.pkl")

y_pred = grid.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

'''plt.scatter(y_pred[:250], y_test[:250], marker='o', color='purple')
m, b = np.polyfit(y_pred, y_test, 1)
plt.plot(y_pred, y_pred * m + b, linestyle='-', color="blue", label="Line of best fit")
plt.plot([0, 25], [0, 25], linestyle='--', color="red", label="Perfect parity")
plt.xlabel("Optimal runs predicted")
plt.ylabel("True runs scored")
plt.legend()
plt.title("Parity plot")
plt.show()'''

'''overs = 20 - X_test[:1000, 1]

plt.scatter(overs, np.abs(y_pred[:1000] - y_test[:1000]))
plt.xlabel("Overs remaining")
plt.xticks([i for i in range(0, 21)])
plt.ylabel("Absolute Error")
plt.title("Error vs Overs Remaining")
plt.show()'''
import pandas as pd

df_test = pd.DataFrame({
    "match_id": match_id_data[test_idx],
    "y_true": y_test,
    "y_pred": y_pred
})

match_error = df_test.groupby("match_id").apply(
    lambda x: np.mean(np.abs(x.y_true - x.y_pred))
)

print(match_error.describe())

m = df_test[df_test["match_id"] == df_test["match_id"].iloc[0]]

ax = plt.figure(figsize=(12, 6)).gca()

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(m["y_true"].values, label="Actual")
plt.plot(m["y_pred"].values, label="Predicted")
plt.legend()
plt.xticks([i for i in range(0, 21)])
plt.xlabel("Overs")
plt.ylabel("Runs")
plt.title("Comparing predicted vs actual runs for Pakistan vs Zimbabwe 2024")
plt.show()


remaining = 200
score = 0
score_total = []
score_match = [0, 12, 16, 25, 32, 46, 59, 68, 74, 83, 87, 99, 104, 125, 135, 151, 164, 176, 180, 197, 203]
score_total.append(0)
np_score = np.array([remaining, 20, 10, 8.0, remaining/20, remaining/20])
np_score = np_score.reshape(1, -1)
for i in range(20):
    pred = grid.predict(np_score)
    pred = round(pred[0])
    print(pred)
    remaining -= pred
    score += pred
    np_score[0][0] = remaining
    np_score[0][1] -= 1
    np_score[0][3] = score/(20-np_score[0][1])
    np_score[0][4] = remaining/np_score[0][1]
    np_score[0][5] = np_score[0][4]-np_score[0][3]
    print(score)
    score_total.append(score)
plt.figure(figsize=(12, 6))
plt.plot([i for i in range(21)], score_total, label="Chase model")
plt.plot([i for i in range(21)], score_match, label="Actual chase")
plt.xticks([i for i in range(0, 21)])
plt.legend()
plt.xlabel("Overs")
plt.ylabel("Runs")
plt.title("Comparing chase model vs actual chase for England vs Pakistan 2022")
plt.show()