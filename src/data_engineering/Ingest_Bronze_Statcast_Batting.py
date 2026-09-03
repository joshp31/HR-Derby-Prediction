from pybaseball import statcast_batter
import pandas as pd

hr_derby_df = pd.read_csv("data/silver/hr derby/HR Derby Participants (2015-2026).csv")

all_statcast = []

for _, row in hr_derby_df.iterrows():

    player_id = row["Player_ID"]
    start_date = row["Year_Start_Date"]
    derby_date = row["Derby_Date"]

    print(f"Pulling {row['Name']} ({row['Year']})")

    batter_stats = statcast_batter(
        start_date,
        derby_date,
        int(player_id)
    )

    # Keep track of who this data belongs to
    batter_stats["Player_ID"] = int(player_id)
    batter_stats["Year"] = row["Year"]

    all_statcast.append(batter_stats)

statcast_df = pd.concat(all_statcast, ignore_index=True)

statcast_df.to_csv("data/bronze/statcast_batting/batter_stats.csv", index=False)