import pandas as pd

def get_spray_direction(row):
    if pd.isna(row["hc_x"]) or pd.isna(row["hc_y"]) or pd.isna(row["stand"]):
        return None

    center_x = 125

    if row["stand"] == "R":
        if row["hc_x"] < center_x - 15:
            return "pull"
        elif row["hc_x"] > center_x + 15:
            return "oppo"
        else:
            return "center"

    elif row["stand"] == "L":
        if row["hc_x"] > center_x + 15:
            return "pull"
        elif row["hc_x"] < center_x - 15:
            return "oppo"
        else:
            return "center"

    return None

silver_statcast_batting_df = pd.read_csv("data/silver/statcast_batting/batter_stats.csv")

silver_statcast_batting_df["spray_direction"] = (
    silver_statcast_batting_df.apply(get_spray_direction, axis=1)
)

truncated_pa_ids = silver_statcast_batting_df.loc[
    silver_statcast_batting_df["events"] == "truncated_pa",
    ["game_pk", "at_bat_number"]
].drop_duplicates()

silver_statcast_batting_df = silver_statcast_batting_df.merge(
    truncated_pa_ids,
    on=["game_pk", "at_bat_number"],
    how="left",
    indicator=True
)

silver_statcast_batting_df = (
    silver_statcast_batting_df[
        silver_statcast_batting_df["_merge"] == "left_only"
    ]
    .drop(columns="_merge")
)

ab_events = [
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "strikeout",
    "grounded_into_double_play",
    "fielders_choice_out",
    "double_play",
    "strikeout_double_play",
    "force_out",
    "field_error",
    "fielders_choice"
]

pa_events = ab_events + [
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt",
    "catcher_interf"
]


gold_statcast_batting_df = (
    silver_statcast_batting_df
    .groupby(["Player_ID", "player_name", "Year"])
    .agg(
        pitches=("Player_ID", "count"),
        plate_appearances=("events", lambda x: x.isin(pa_events).sum()),
        at_bats=("events", lambda x: x.isin(ab_events).sum()),
        walks=("events", lambda x: (x == "walk").sum()),
        intentional_walks=("events", lambda x: (x == "intent_walk").sum()),
        strikeouts=("events", lambda x: (x.isin(["strikeout", "strikeout_double_play"])).sum()),
        hit_by_pitch=("events", lambda x: (x == "hit_by_pitch").sum()),
        singles=("events", lambda x: (x == "single").sum()),
        doubles=("events", lambda x: (x == "double").sum()),
        triples=("events", lambda x: (x == "triple").sum()),
        home_runs=("events", lambda x: (x == "home_run").sum()),
        sac_flies=("events", lambda x: (x.isin(["sac_fly", "sac_fly_double_play"])).sum()),
        Avg_EV=("launch_speed", "mean"),
        Max_EV=("launch_speed", "max"),
        EV_95=("launch_speed", lambda x: x.quantile(0.95)),
        Avg_LA=("launch_angle", "mean"),
        Median_LA=("launch_angle", "median"),
        Std_LA=("launch_angle", "std"),
        Avg_bat_speed=("bat_speed", "mean"),
        Max_bat_speed=("bat_speed", "max"),
        Avg_attack_angle=("attack_angle", "mean"),
        Median_attack_angle=("attack_angle", "median"),
        Std_attack_angle=("attack_angle", "std"),
        Avg_attack_direction=("attack_direction", "mean"),
        Median_attack_direction=("attack_direction", "median"),
        Std_attack_direction=("attack_direction", "std"),
        Avg_swing_path_tilt=("swing_path_tilt", "mean"),
        Median_swing_path_tilt=("swing_path_tilt", "median"),
        Std_swing_path_tilt=("swing_path_tilt", "std"),
        Avg_swing_length=("swing_length", "mean"),
        Median_swing_length=("swing_length", "median"),
        Std_swing_length=("swing_length", "std"),
        sweet_spot=("launch_angle",lambda x: x.between(8,32).sum()),
        batted_balls=("launch_speed", lambda x: x.notna().sum()),
        hard_hits=("launch_speed", lambda x: (x >= 95).sum()),
        barrels=("launch_speed_angle", lambda x: (x == 6).sum()),
        ground_balls=("launch_angle", lambda x: (x < 10).sum()),
        air_balls=("launch_angle", lambda x: x.between(10, 50, inclusive="left").sum()),
        popups=("launch_angle", lambda x: (x > 50).sum()),
        pulls=("spray_direction", lambda x: (x == "pull").sum()),
        centers=("spray_direction", lambda x: (x == "center").sum()),
        oppos=("spray_direction", lambda x: (x == "oppo").sum()),
        spray_balls=("spray_direction", lambda x: x.notna().sum())
    )
    .reset_index()
)

gold_statcast_batting_df["hits"] = gold_statcast_batting_df["singles"] + gold_statcast_batting_df["doubles"] + gold_statcast_batting_df["triples"] + gold_statcast_batting_df["home_runs"]

gold_statcast_batting_df["batting_average"] = gold_statcast_batting_df["hits"] / gold_statcast_batting_df["at_bats"]

gold_statcast_batting_df["total_bases"] = (
    gold_statcast_batting_df["singles"]
    + 2 * gold_statcast_batting_df["doubles"]
    + 3 * gold_statcast_batting_df["triples"]
    + 4 * gold_statcast_batting_df["home_runs"]
)

gold_statcast_batting_df["slugging_percentage"] = (
    gold_statcast_batting_df["total_bases"]
    / gold_statcast_batting_df["at_bats"]
)

gold_statcast_batting_df["on_base_percentage"] = (
    (gold_statcast_batting_df["hits"] + gold_statcast_batting_df["walks"] + gold_statcast_batting_df["intentional_walks"] + gold_statcast_batting_df["hit_by_pitch"]) /
    (gold_statcast_batting_df["at_bats"] + gold_statcast_batting_df["walks"] + gold_statcast_batting_df["intentional_walks"] + gold_statcast_batting_df["hit_by_pitch"] + gold_statcast_batting_df["sac_flies"])
)

gold_statcast_batting_df["ops"] = gold_statcast_batting_df["on_base_percentage"] + gold_statcast_batting_df["slugging_percentage"]

gold_statcast_batting_df["iso"] = gold_statcast_batting_df["slugging_percentage"] - gold_statcast_batting_df["batting_average"]

gold_statcast_batting_df["ab/hr"] = gold_statcast_batting_df["at_bats"] / gold_statcast_batting_df["home_runs"]

gold_statcast_batting_df["pa/hr"] = gold_statcast_batting_df["plate_appearances"] / gold_statcast_batting_df["home_runs"]

gold_statcast_batting_df["bb%"] = (gold_statcast_batting_df["walks"] + gold_statcast_batting_df["intentional_walks"]) / gold_statcast_batting_df["plate_appearances"]

gold_statcast_batting_df["k%"] = gold_statcast_batting_df["strikeouts"] / gold_statcast_batting_df["plate_appearances"]

gold_statcast_batting_df["xbh%"] = (gold_statcast_batting_df["doubles"] + gold_statcast_batting_df["triples"] + gold_statcast_batting_df["home_runs"]) / gold_statcast_batting_df["plate_appearances"]

gold_statcast_batting_df["total_bases_per_pa"] = gold_statcast_batting_df["total_bases"] / gold_statcast_batting_df["plate_appearances"]

gold_statcast_batting_df["hard_hit%"] = gold_statcast_batting_df["hard_hits"] / gold_statcast_batting_df["batted_balls"]

gold_statcast_batting_df["barrel%"] = gold_statcast_batting_df["barrels"] / gold_statcast_batting_df["batted_balls"]

gold_statcast_batting_df["sweet_spot%"] = gold_statcast_batting_df["sweet_spot"] / gold_statcast_batting_df["batted_balls"]

gold_statcast_batting_df["air_ball%"] = gold_statcast_batting_df["air_balls"] / gold_statcast_batting_df["batted_balls"]

gold_statcast_batting_df["ground_ball%"] = gold_statcast_batting_df["ground_balls"] / gold_statcast_batting_df["batted_balls"]

gold_statcast_batting_df["popup%"] = gold_statcast_batting_df["popups"] / gold_statcast_batting_df["batted_balls"]

gold_statcast_batting_df["pull_percentage"] = (gold_statcast_batting_df["pulls"] / gold_statcast_batting_df["spray_balls"])

gold_statcast_batting_df["center_percentage"] = (gold_statcast_batting_df["centers"] / gold_statcast_batting_df["spray_balls"])

gold_statcast_batting_df["oppo_percentage"] = (gold_statcast_batting_df["oppos"] / gold_statcast_batting_df["spray_balls"])

gold_statcast_batting_df = gold_statcast_batting_df.dropna(axis=1)

derby_df = pd.read_csv("data/silver/hr derby/HR Derby Participants (2015-2026).csv")

merged_df = gold_statcast_batting_df.merge(
    derby_df[["Player_ID", "Year", "Round_Reached"]],
    on=["Player_ID", "Year"],
    how="inner"
)

merged_df_pre_2026 = merged_df[merged_df["Year"] < 2026].copy()
merged_df_2026 = merged_df[merged_df["Year"] == 2026].copy()

merged_df_pre_2026.to_csv("data/gold/statcast_batting/batter_stats_pre_2026.csv", index=False)
merged_df_2026.to_csv("data/gold/statcast_batting/batter_stats_2026.csv", index=False)
merged_df.to_csv("data/gold/statcast_batting/batter_stats.csv", index=False)