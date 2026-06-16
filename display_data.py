import joblib
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


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

grid = joblib.load("cricket_win_prob_model.pkl")

# Split the dataset into training and test sets'''
pred_probs = grid.predict_proba(X_test)[:, 1]
'''prob_true, prob_pred = calibration_curve(
    y_test,
    pred_probs,
    n_bins=10,
    strategy='uniform'
)

plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.legend()
plt.title("Calibration Curve")
plt.show()'''

losses = (y_test - pred_probs)**2
brier_score = losses.mean()  # average over all samples
print("Brier score:", brier_score)
plt.figure(figsize=(12, 8))
true_vals = [[] for _ in range(9)]
for i in range (len(pred_probs)):
    idx = min(int(pred_probs[i] * 9), 8)
    true_vals[idx].append(y_test[i])
graph_diff = []
graph_count = []
graph_true = []
for i in range(len(true_vals)):
    graph_true.append(np.mean(true_vals[i]))
    graph_diff.append(abs(np.mean(true_vals[i]) - (2 * i + 1)/18))
    graph_count.append(len(true_vals[i]))

bin_diff = np.array(graph_diff).reshape(3, 3)
bin_count = np.array(graph_count).reshape(3, 3)
bin_true = np.array(graph_true).reshape(3, 3)
heatmap = 1 - bin_diff
im = plt.imshow(heatmap, cmap="viridis", vmin=0.9, vmax=1)
plt.colorbar(label="Calibration quality (1 = best)")
font = {'family': 'helvetica',
        'color':  'black',
        'weight': 'normal',
        'size': 11,
        }
for i in range(0, 3):
    for j in range(0, 3):
        val = 3 * i + j
        plt.text(j, i, 
                 f"Exp: {round((val*2 + 1)/18, 2)}\n"
                 f"True: {round(bin_true[i][j], 2)}\n"
                 f"Count: {bin_count[i][j]}", ha='center', va='center', fontdict=font)
plt.title("3x3 Calibration Grid")
plt.axis("off")
plt.show()
'''#over, runs, wickets, batter 1 score, batter 2 score
#runs remaining, overs remaining, wickets remaining, crr, nrr
run_chase = [[0, 0, 0, 0, 0], [1, 6, 0, 1, 5], [2, 13, 0, 4, 6], [3, 31, 0, 20, 7], [4, 41, 0, 22, 15], 
             [5, 49, 2, 7, 1], [6, 56, 2, 13, 2], [7, 68, 2, 19, 8], [8, 72, 2, 21, 10], [9, 84, 3, 23, 8],
             [10, 85, 3, 23, 9], [11, 90, 3, 23, 13], [12, 96, 4, 28, 1], [13, 104, 5, 33, 0],
             [14, 119, 5, 48, 0], [15, 127, 6, 1, 1], [16, 128, 7, 0, 2], [17, 139, 7, 0, 13],
             [18, 152, 7, 0, 26], [19, 168, 7, 0, 42], [20, 182, 7, 1, 54]]
for i in range(len(run_chase)):
    run_chase[i].append(182 - run_chase[i][1])
    run_chase[i].append(20 - run_chase[i][0])
    run_chase[i].append(10 - run_chase[i][2])
    run_chase[i].append(run_chase[i][1] / run_chase[i][0] if run_chase[i][0] != 0 else 0)
    run_chase[i].append(run_chase[i][5] / run_chase[i][6] if run_chase[i][6] != 0 and run_chase[i][5] > 0 else 0)

run_chase = np.array(run_chase)

overs = [i for i in range(21)]
pred_prob_match = grid.predict_proba(run_chase)[:, 1] 
plt.figure(figsize=(12, 6))
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.plot(overs, pred_prob_match, marker = "o", clip_on=False)
plt.xticks(overs)
plt.xlabel("Chasing Over")
plt.ylabel("Win probability for LSG")
plt.title("Win probability for KKR vs LSG on April 9")
plt.show()'''
