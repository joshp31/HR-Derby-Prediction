import pandas as pd
import matplotlib.pyplot as plt

results_path = (
    "results/accuracy_by_year.csv"
)

df = pd.read_csv(results_path)

df = df.sort_values("year")

plt.figure(figsize=(10, 7))

plt.plot(
    df["year"],
    df["accuracy"],
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Accuracy")
plt.title(
    "Neural Network Accuracy by Year"
)

plt.ylim(0, 1)
plt.xticks(df["year"])

for _, row in df.iterrows():
    plt.text(
        row["year"],
        row["accuracy"] + 0.03,
        f"{row['accuracy']:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/data_visualizations/accuracy_by_year.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()