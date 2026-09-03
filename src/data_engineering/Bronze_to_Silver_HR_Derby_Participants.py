from pybaseball import playerid_lookup
import pandas as pd

hr_derby_df = pd.read_csv('data/bronze/hr derby/HR Derby Participants (2015-2026).csv')

player_ids = []

for _, row in hr_derby_df.iterrows():
    name = row["Name"]
    derby_year = row["Year"]

    first_name, last_name = name.split(" ", 1)

    result = playerid_lookup(last_name, first_name)

    if len(result) == 1:
        player_ids.append(result.iloc[0]["key_mlbam"])

    elif len(result) > 1:
        valid_players = result[
            (result["mlb_played_first"] <= derby_year) &
            (result["mlb_played_last"] >= derby_year)
        ]

        if len(valid_players) == 1:
            player_ids.append(valid_players.iloc[0]["key_mlbam"])

        else:
            print(f"Could not uniquely identify {name}")
            print(result)
            player_ids.append(None)

    else:
        print(f"No match found for {name}")
        player_ids.append(None)

hr_derby_df["Derby_Date"] = pd.to_datetime(hr_derby_df["Derby_Date"])

hr_derby_df["Year_Start_Date"] = pd.to_datetime(hr_derby_df["Year"].astype(str) + "-01-01")

hr_derby_df["Player_ID"] = player_ids

hr_derby_df.to_csv("data/silver/hr derby/HR Derby Participants (2015-2026).csv", index=False)