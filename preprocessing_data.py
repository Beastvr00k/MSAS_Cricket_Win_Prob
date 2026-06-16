import numpy as np
import glob
import pandas as pd
import hashlib

X_data = []
y_data = []
weight_data = []
match_id_data = []

def weight(team_1_score, team_2_score, overs_remaining=0):
    if overs_remaining != 0:
        team_2_score = round((team_2_score * 20)/(20-overs_remaining))
    return 2 ** (-5 * (abs(team_2_score - team_1_score))/team_1_score)

for filename in glob.glob("./all_csv/*.csv"):
    df = pd.read_csv(filename)
    df = df.dropna().drop(columns=["innings"])
    match_id = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
    df['current_run_rate'] = df.apply(lambda row: row.total_runs / row.over if row.over != 0 else 0, axis=1)
    df['required_run_rate'] = df.apply(lambda row: row.runs_remaining / row.overs_remaining if row.overs_remaining != 0 and row.runs_remaining > 0 else 0, axis=1)
    df['runs_scored'] = df['runs_remaining'] - df['runs_remaining'].shift(-1)
    df['runs_scored'] = df['runs_scored'].fillna(0)
    if(df.iloc[-1]["runs_remaining"] > 1):
        win_cond = 0
        weight_game = weight(df.iloc[0]["runs_remaining"] - 1, df.iloc[-1]["total_runs"])
    elif(df.iloc[-1]["runs_remaining"] <= 0):
        win_cond = 1
        weight_game = weight(df.iloc[0]["runs_remaining"] - 1, df.iloc[-1]["total_runs"], df.iloc[-1]["overs_remaining"])
    else:
        continue
    for i in range(0, len(df)):
        X_data.append(df.iloc[i].values)
        y_data.append(win_cond)
        weight_data.append(weight_game)
        match_id_data.append(match_id)
    print(len(X_data))

X_data = np.array(X_data)
y_data = np.array(y_data)
weight_data = np.array(weight_data)
match_id_data = np.array(match_id_data)

np.save("X_data.npy", X_data)
np.save("y_data.npy", y_data)
np.save("weight_data.npy", weight_data)
np.save("match_id_data.npy", match_id_data)