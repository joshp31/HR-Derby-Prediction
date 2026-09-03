import pandas as pd
import matplotlib.pyplot as plt

results_path = "results/overall_model_results.csv"
df = pd.read_csv(results_path)

df = df.sort_values("champion_accuracy", ascending=True)

plt.figure(figsize=(10, 9))

plt.barh(
    df["model"], 
    df["champion_accuracy"],
    height=0.5
)

plt.xlabel("Champion Accuracy")
plt.ylabel("Model")
plt.title("Model Champion Accuracy Comparison")

for i, accuracy in enumerate(df["champion_accuracy"]):
    plt.text(
        accuracy + 0.005,
        i,
        f"{accuracy:.3f}",
        va="center"
    )

plt.xlim(0, 1)
plt.tight_layout()

output_path = "results/data_visualizations/model_champion_accuracy_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")