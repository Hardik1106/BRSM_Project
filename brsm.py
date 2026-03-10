import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


base_folder = r"C:\Users\mavur_crvzl5q\Downloads\360 Videos VR project\data\headtracking-data"
survey_file = r"C:\Users\mavur_crvzl5q\Downloads\360 Videos VR project\data\data.xlsx"

videos = ["v1","v2","v3","v4","v5"]

rows = []


for video in videos:

    folder = os.path.join(base_folder, video)
    files = glob.glob(os.path.join(folder, "*.csv"))

    for file in files:

        try:

            df = pd.read_csv(file, engine="python", on_bad_lines="skip")

            mean_ang_vel = df["RotationSpeedTotal"].astype(float).mean()
            rot_change_y_sd = df["RotationChangeY"].astype(float).std()

            filename = os.path.basename(file).split(".")[0]

            timestamp = filename.split("_")[-1]

            rows.append({
                "timestamp": timestamp,
                "mean_ang_vel": mean_ang_vel,
                "rot_change_y_sd": rot_change_y_sd
            })

        except Exception as e:
            print("Error reading:", file)
            print(e)


movement = pd.DataFrame(rows)



movement_avg = (
    movement.groupby("timestamp")[["mean_ang_vel","rot_change_y_sd"]]
    .mean()
    .reset_index()
)

print("\nHead movement summary:")
print(movement_avg.head())


survey = pd.read_excel(survey_file)

survey["TIME_start"] = pd.to_datetime(survey["TIME_start"], errors="coerce")

survey["timestamp"] = survey["TIME_start"].dt.strftime("%Y%m%d%H%M%S")

print("\nSurvey timestamps:")
print(survey["timestamp"].head())



movement_avg["timestamp14"] = movement_avg["timestamp"].str[:14]
movement_avg["time"] = pd.to_datetime(movement_avg["timestamp14"], format="%Y%m%d%H%M%S")

survey["time"] = pd.to_datetime(survey["TIME_start"], errors="coerce")

movement_avg = movement_avg.sort_values("time")
survey = survey.sort_values("time")


merged = pd.merge_asof(
    movement_avg,
    survey[["time", "score_phq"]],
    on="time",
    direction="nearest",
    tolerance=pd.Timedelta("10min")  
)

merged = merged.dropna(subset=["score_phq"])

print("\nMerged dataset:")
print(merged.head())

plt.figure(figsize=(7,5))

plt.scatter(
    merged["score_phq"],
    merged["mean_ang_vel"]
)

m,b = np.polyfit(merged["score_phq"], merged["mean_ang_vel"],1)

plt.plot(
    merged["score_phq"],
    m*merged["score_phq"] + b
)

plt.xlabel("PHQ-9 Score")
plt.ylabel("Mean Angular Velocity")
plt.title("PHQ-9 vs Head Movement")

plt.show()
