import pandas as pd

statcast_batting_df = pd.read_csv("data/gold/statcast_batting/batter_stats.csv")

relative_stats = [
    "plate_appearances",

    "home_runs",

    "Avg_EV",
    "Max_EV",
    "EV_95",
    "Avg_LA",
    "Median_LA",
    "Std_LA",

    "ops",
    "iso",
    "ab/hr",
    "bb%",
    "k%",
    "xbh%",
    "total_bases_per_pa",

    "hard_hit%",
    "barrel%",
    "sweet_spot%",

    "air_ball%",
    "ground_ball%",
    "popup%",

    "pull_percentage",
    "center_percentage"
]

for stat in relative_stats:
    statcast_batting_df[f"{stat}_above_avg"] = (
        statcast_batting_df[stat]
        - statcast_batting_df.groupby("Year")[stat].transform("mean")
    )

statcast_batting_pre_2026_df = statcast_batting_df[statcast_batting_df["Year"] < 2026].copy()
statcast_batting_2026_df = statcast_batting_df[statcast_batting_df["Year"] == 2026].copy()

statcast_batting_pre_2026_df.to_csv("data/gold/statcast_batting_w_relative_features/batter_stats_relative_pre_2026.csv", index=False)
statcast_batting_2026_df.to_csv("data/gold/statcast_batting_w_relative_features/batter_stats_relative_2026.csv", index=False)
statcast_batting_df.to_csv("data/gold/statcast_batting_w_relative_features/batter_stats_relative.csv", index=False)