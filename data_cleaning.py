import json
import os
import copy
import pandas as pd

folder = "all_json"
for filename in os.listdir(folder):
    filename = "1487715.json"
    filepath = os.path.join(folder, filename)
    if not filepath.lower().endswith(".json"):
        continue
    try:
        with open(filepath) as f:
            data = json.load(f)
            print(f"Opened {filepath}")
            first_inning_overs_list = [] #list of all overs
            if data["info"]["match_type"] != "T20":
                os.remove(filepath)
                continue
            overs = data["innings"][0]["overs"]
            first_ball = overs[0]['deliveries'][0]
            wicket_bool = False
            ball_list = [{first_ball['batter']: 0, first_ball['non_striker']: 0}, 0, 0, 0] #runs, overs, wickets"
            first_inning_overs_list.append(copy.deepcopy(ball_list))
            for over in overs:
                for delivery in over["deliveries"]:
                    if(wicket_bool):
                        if(delivery["batter"] not in ball_list[0]):
                            ball_list[0][delivery["batter"]] = 0
                        if(delivery["non_striker"] not in ball_list[0]):
                            ball_list[0][delivery["non_striker"]] = 0
                    ball_list[0][delivery["batter"]] += delivery["runs"]["batter"]
                    ball_list[1] += delivery["runs"]["total"]
                    if "wickets" in delivery:
                        if(len(delivery["wickets"]) == 1):
                            ball_list[0].pop(delivery["wickets"][0]["player_out"])
                        else:
                            ball_list[0] = {}
                        ball_list[3] += len(delivery["wickets"])
                    wicket_bool = "wickets" in delivery
                ball_list[2] += 1
                first_inning_overs_list.append(copy.deepcopy(ball_list))
            if(ball_list[3] < 10 and len(overs) < 20):
                os.remove(filepath)
                continue
            total_runs = ball_list[1]
            second_inning_overs_list = [] #list of all overs
            overs = data["innings"][1]["overs"]
            first_ball = overs[0]['deliveries'][0]
            wicket_bool = False
            ball_list = [{first_ball['batter']: 0, first_ball['non_striker']: 0}, 0, 0, 0, total_runs + 1, 20, 10] #runs, overs, wickets, runs_remaining, overs_remaining, wickets_remaining"
            second_inning_overs_list.append(copy.deepcopy(ball_list))
            for over in overs:
                for delivery in over["deliveries"]:
                    if(wicket_bool):
                        if(delivery["batter"] not in ball_list[0]):
                            ball_list[0][delivery["batter"]] = 0
                        if(delivery["non_striker"] not in ball_list[0]):
                            ball_list[0][delivery["non_striker"]] = 0
                    ball_list[0][delivery["batter"]] += delivery["runs"]["batter"]
                    ball_list[1] += delivery["runs"]["total"]
                    ball_list[4] -= delivery["runs"]["total"]
                    if "wickets" in delivery:
                        if(len(delivery["wickets"]) == 1):
                            ball_list[0].pop(delivery["wickets"][0]["player_out"])
                        else:
                            ball_list[0] = {}
                        ball_list[3] += len(delivery["wickets"])
                        ball_list[6] -= len(delivery["wickets"])
                    wicket_bool = "wickets" in delivery
                ball_list[2] += 1
                ball_list[5] -= 1
                print(ball_list)
                second_inning_overs_list.append(copy.deepcopy(ball_list))
            if(ball_list[3] < 10 and len(overs) < 20 and ball_list[4] > 0):
                os.remove(filepath)
                continue
        combined_list = []
        for over in first_inning_overs_list:
            row = {"innings": 1, "over": over[2], "total_runs": over[1], "wickets": over[3]}
            batters = list(over[0].values())  # just the runs
            row["batter_1_runs"] = batters[0] if len(batters) > 0 else 0
            row["batter_2_runs"] = batters[1] if len(batters) > 1 else 0
            combined_list.append(row)
        for over in second_inning_overs_list:
            row = {"innings": 2, "over": over[2], "total_runs": over[1], "wickets": over[3], 
                    "runs_remaining": over[4], "overs_remaining": over[5], "wickets_remaining": over[6]}
            batters = list(over[0].values())
            row["batter_1_runs"] = batters[0] if len(batters) > 0 else 0
            row["batter_2_runs"] = batters[1] if len(batters) > 1 else 0
            combined_list.append(row)
        df = pd.DataFrame(combined_list)
        output_file = f"{filename}.csv"
        df.to_csv(output_file, index=False)
    
    except Exception as e:
        print(f"Skipping {filepath}: {e}")



    

