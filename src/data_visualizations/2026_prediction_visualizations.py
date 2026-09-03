import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

PRE_2026_PATH = "data/gold/statcast_batting_w_relative_features/batter_stats_relative_pre_2026.csv"
DATA_2026_PATH = "data/gold/statcast_batting_w_relative_features/batter_stats_relative_2026.csv"

OUTPUT_ACTUAL_PREDICTED = "results/data_visualizations/2026_actual_vs_predicted.png"
OUTPUT_PROBABILITY_PLOT = "results/data_visualizations/2026_round_probabilities.png"

pre_2026 = pd.read_csv(PRE_2026_PATH)
data_2026 = pd.read_csv(DATA_2026_PATH)

features = [
    "Max_EV",
    "EV_95_above_avg",
    "sweet_spot%"
]

X_train = pre_2026[features]
y_train = pre_2026["Round_Reached"]

X_2026 = data_2026[features]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        alpha=0.1,
        hidden_layer_sizes=(5, 5),
        max_iter=5000,
        random_state=31
    ))
])

model.fit(X_train, y_train)

def assign_rounds(probabilities):
    """
    Assign valid Home Run Derby rounds to the 8 participants.

    Round structure:
        4 players eliminated in Round 1
        2 players eliminated in Round 2
        1 player eliminated in Round 3
        1 Champion

    The assignment maximizes the total log probability.
    """

    n_players = len(probabilities)

    if n_players != 8:
        raise ValueError(
            f"Expected 8 players for the Home Run Derby, "
            f"but received {n_players}."
        )

    best_score = -np.inf
    best_assignment = None

    for champion in range(n_players):

        for finalist in range(n_players):

            if finalist == champion:
                continue

            remaining_after_champion = [
                i for i in range(n_players)
                if i not in [champion, finalist]
            ]

            for semifinalists in __import__("itertools").combinations(
                remaining_after_champion, 2
            ):

                assignment = np.ones(n_players, dtype=int)

                assignment[champion] = 4
                assignment[finalist] = 3

                for player in semifinalists:
                    assignment[player] = 2

                score = 0

                for player in range(n_players):
                    round_index = assignment[player] - 1
                    probability = probabilities[player, round_index]

                    probability = max(probability, 1e-12)

                    score += np.log(probability)

                if score > best_score:
                    best_score = score
                    best_assignment = assignment.copy()

    return best_assignment

probabilities = model.predict_proba(X_2026)

predicted_rounds = assign_rounds(probabilities)

prediction_df = data_2026[
    ["player_name", "Round_Reached"]
].copy()

prediction_df["Predicted_Round"] = predicted_rounds

prediction_df["Correct"] = (
    prediction_df["Round_Reached"]
    == prediction_df["Predicted_Round"]
)

for _, row in prediction_df.iterrows():

    actual = int(row["Round_Reached"])
    predicted = int(row["Predicted_Round"])

accuracy = prediction_df["Correct"].mean()

plot_df = prediction_df.sort_values("Round_Reached").copy()

y = np.arange(len(plot_df))

bar_height = 0.35

plt.figure(figsize=(10, 7))

actual_bars = plt.barh(
    y - bar_height / 2,
    plot_df["Round_Reached"],
    height=bar_height,
    label="Actual Round"
)

predicted_bars = plt.barh(
    y + bar_height / 2,
    plot_df["Predicted_Round"],
    height=bar_height,
    label="Predicted Round"
)

plt.yticks(
    y,
    plot_df["player_name"]
)

plt.xlabel("Round Reached")
plt.ylabel("Player")

plt.title(
    "2026 Home Run Derby: Actual vs. Predicted Round\n",
    fontweight="bold"
)

plt.xticks(
    [1, 2, 3, 4],
    ["Round 1", "Round 2", "Round 3", "Champion"]
)

plt.xlim(0, 4.5)

for bar in actual_bars:

    round_number = int(bar.get_width())

    plt.text(
        bar.get_width() + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"R{round_number}",
        va="center"
    )


for bar in predicted_bars:

    round_number = int(bar.get_width())

    plt.text(
        bar.get_width() + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"R{round_number}",
        va="center"
    )


plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ACTUAL_PREDICTED,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

probability_df = prediction_df[
    [
        "player_name",
        "Round_Reached",
        "Predicted_Round"
    ]
].copy()

probability_df["Round_1_Probability"] = probabilities[:, 0] * 100
probability_df["Round_2_Probability"] = probabilities[:, 1] * 100
probability_df["Round_3_Probability"] = probabilities[:, 2] * 100
probability_df["Champion_Probability"] = probabilities[:, 3] * 100

probability_plot_df = probability_df.sort_values(
    "Round_Reached"
).reset_index(drop=True)

colors = [
    "#2563EB",
    "#F59E0B",
    "#10B981",
    "#E11D48"
]

round_labels = [
    "Round 1",
    "Round 2",
    "Round 3",
    "Champion"
]

probability_columns = [
    "Round_1_Probability",
    "Round_2_Probability",
    "Round_3_Probability",
    "Champion_Probability"
]

n_players = len(probability_plot_df)

y = np.arange(n_players) * 1.35

fig, ax = plt.subplots(
    figsize=(14, 10)
)

left = np.zeros(n_players)

for i, (column, label, color) in enumerate(
    zip(
        probability_columns,
        round_labels,
        colors
    )
):

    values = probability_plot_df[column].values

    ax.barh(
        y,
        values,
        left=left,
        height=0.62,
        color=color,
        label=label
    )

    left += values

for row_index in range(n_players):

    cumulative = 0

    for segment_index, column in enumerate(probability_columns):

        value = probability_plot_df.loc[
            row_index,
            column
        ]

        center = cumulative + value / 2

        if value >= 6:

            ax.text(
                center,
                y[row_index],
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white"
            )

        else:

            if segment_index % 2 == 0:
                annotation_y = y[row_index] - 0.47
            else:
                annotation_y = y[row_index] + 0.47

            ax.annotate(
                f"{value:.1f}%",
                xy=(center, y[row_index]),
                xytext=(center, annotation_y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=0.8
                )
            )

        cumulative += value

for row_index in range(n_players):

    actual_round = int(
        probability_plot_df.loc[
            row_index,
            "Round_Reached"
        ]
    )

    predicted_round = int(
        probability_plot_df.loc[
            row_index,
            "Predicted_Round"
        ]
    )

    actual_x = (
        probability_plot_df.loc[
            row_index,
            probability_columns[:actual_round]
        ].sum()
        - probability_plot_df.loc[
            row_index,
            probability_columns[actual_round - 1]
        ] / 2
    )

    predicted_x = (
        probability_plot_df.loc[
            row_index,
            probability_columns[:predicted_round]
        ].sum()
        - probability_plot_df.loc[
            row_index,
            probability_columns[predicted_round - 1]
        ] / 2
    )

    ax.scatter(
        actual_x,
        y[row_index] - 0.48,
        marker="D",
        s=85,
        color="black",
        edgecolor="white",
        linewidth=1.2,
        zorder=5
    )

    ax.scatter(
        predicted_x,
        y[row_index] + 0.48,
        marker="o",
        s=95,
        facecolors="white",
        edgecolors="black",
        linewidth=2,
        zorder=5
    )

ax.set_xlim(0, 100)

ax.set_yticks(y)

ax.set_yticklabels(
    probability_plot_df["player_name"],
    fontsize=10
)

ax.set_xlabel(
    "Probability (%)",
    fontsize=11
)

ax.set_ylabel(
    "Player",
    fontsize=11
)

ax.set_title(
    "2026 Home Run Derby: Predicted Round Probabilities",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.set_xticks(
    [0, 20, 40, 60, 80, 100]
)

ax.set_xticklabels(
    ["0%", "20%", "40%", "60%", "80%", "100%"]
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

ax.set_axisbelow(True)

ax.invert_yaxis()

probability_legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False,
    fontsize=10
)

ax.add_artist(probability_legend)

symbol_legend = [
    Line2D(
        [0],
        [0],
        marker="D",
        color="black",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=8,
        linewidth=0,
        label="Actual round"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=9,
        linewidth=0,
        label="Predicted round"
    )
]

ax.legend(
    handles=symbol_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=2,
    frameon=False,
    fontsize=10
)

plt.subplots_adjust(
    left=0.18,
    right=0.98,
    top=0.91,
    bottom=0.20
)

plt.savefig(
    OUTPUT_PROBABILITY_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()