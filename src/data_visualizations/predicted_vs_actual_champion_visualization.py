import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

results_path = "results/champion_predictions.csv"

output_path = (
    "results/data_visualizations/"
    "predicted_vs_actual_champion.png"
)

df = pd.read_csv(results_path)

df = df.sort_values("Year").reset_index(drop=True)

df["Correct"] = (
    df["Correct"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

accuracy = df["Correct"].mean()
correct_count = int(df["Correct"].sum())
total_count = len(df)

sns.set_theme(
    style="white",
    context="talk"
)

actual_color = "#2563EB"
predicted_correct_color = "#10B981"
predicted_incorrect_color = "#EF4444"

correct_light = "#ECFDF5"
incorrect_light = "#FEF2F2"

text_dark = "#111827"
text_gray = "#6B7280"
border_gray = "#D1D5DB"

fig, ax = plt.subplots(
    figsize=(15, 11)
)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_xlim(0, 15)
ax.set_ylim(0, len(df) * 1.35 + 3.8)

ax.axis("off")

ax.text(
    7.5,
    len(df) * 1.35 + 3.10,
    "Home Run Derby Champion Predictions",
    ha="center",
    va="center",
    fontsize=25,
    fontweight="bold",
    color=text_dark
)

ax.text(
    7.5,
    len(df) * 1.35 + 2.50,
    "Actual champions vs. model predictions",
    ha="center",
    va="center",
    fontsize=12,
    color=text_gray
)

accuracy_x = 5.0
accuracy_y = len(df) * 1.35 + 1.25
accuracy_width = 5.0
accuracy_height = 0.75

accuracy_box = FancyBboxPatch(
    (
        accuracy_x,
        accuracy_y
    ),
    accuracy_width,
    accuracy_height,
    boxstyle="round,pad=0.03,rounding_size=0.12",
    linewidth=0,
    facecolor="#111827"
)

ax.add_patch(accuracy_box)

ax.text(
    accuracy_x + 2.5,
    accuracy_y + 0.48,
    f"{accuracy:.1%}",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="white"
)

ax.text(
    accuracy_x + 2.5,
    accuracy_y + 0.15,
    f"Champion Accuracy  •  {correct_count}/{total_count} correct",
    ha="center",
    va="center",
    fontsize=9,
    color="white"
)

header_y = len(df) * 1.35 + 0.65

ax.text(
    3.55,
    header_y,
    "ACTUAL CHAMPION",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=text_gray
)

ax.text(
    9.05,
    header_y,
    "MODEL PREDICTION",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=text_gray
)

start_y = len(df) * 1.35 - 0.10

card_height = 1.05

for i, row in df.iterrows():

    y = start_y - i * 1.35

    actual = row["Actual_Champion"]
    predicted = row["Predicted_Champion"]
    year = int(row["Year"])
    correct = row["Correct"]

    card = FancyBboxPatch(
        (0.45, y - 0.52),
        14.1,
        card_height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1,
        edgecolor="#E5E7EB",
        facecolor="#F9FAFB"
    )

    ax.add_patch(card)

    ax.text(
        0.95,
        y,
        str(year),
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=text_dark
    )

    actual_box = FancyBboxPatch(
        (2.05, y - 0.32),
        3.0,
        0.64,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0,
        facecolor=actual_color
    )

    ax.add_patch(actual_box)

    ax.text(
        3.55,
        y,
        actual,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white"
    )

    ax.text(
        6.25,
        y,
        "→",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color="#9CA3AF"
    )

    if correct:

        prediction_color = predicted_correct_color
        prediction_background = correct_light

    else:

        prediction_color = predicted_incorrect_color
        prediction_background = incorrect_light


    predicted_box = FancyBboxPatch(
        (7.55, y - 0.32),
        3.0,
        0.64,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2,
        edgecolor=prediction_color,
        facecolor=prediction_background
    )

    ax.add_patch(predicted_box)

    ax.text(
        9.05,
        y,
        predicted,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=prediction_color
    )

    if correct:

        badge_text = "CORRECT"
        badge_color = predicted_correct_color

    else:

        badge_text = "INCORRECT"
        badge_color = predicted_incorrect_color


    badge = FancyBboxPatch(
        (11.15, y - 0.25),
        2.65,
        0.50,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0,
        facecolor=badge_color
    )

    ax.add_patch(badge)

    ax.text(
        12.475,
        y,
        badge_text,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white"
    )

legend_y = -0.15

actual_legend = FancyBboxPatch(
    (4.35, legend_y - 0.20),
    0.45,
    0.40,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=0,
    facecolor=actual_color
)

ax.add_patch(actual_legend)

ax.text(
    5.00,
    legend_y,
    "Actual champion",
    ha="left",
    va="center",
    fontsize=9,
    color=text_gray
)

correct_legend = FancyBboxPatch(
    (7.35, legend_y - 0.20),
    0.45,
    0.40,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=0,
    facecolor=predicted_correct_color
)

ax.add_patch(correct_legend)

ax.text(
    8.00,
    legend_y,
    "Correct prediction",
    ha="left",
    va="center",
    fontsize=9,
    color=text_gray
)

incorrect_legend = FancyBboxPatch(
    (10.55, legend_y - 0.20),
    0.45,
    0.40,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=0,
    facecolor=predicted_incorrect_color
)

ax.add_patch(incorrect_legend)

ax.text(
    11.20,
    legend_y,
    "Incorrect prediction",
    ha="left",
    va="center",
    fontsize=9,
    color=text_gray
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()