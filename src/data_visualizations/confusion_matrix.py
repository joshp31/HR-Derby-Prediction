import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions_path = (
    "results/model_predictions.csv"
)

df = pd.read_csv(predictions_path)

actual = df["actual_round"]
predicted = df["predicted_round"]

labels = [1, 2, 3, 4]
display_labels = [
    "Round 1",
    "Round 2",
    "Round 3",
    "Champion"
]

cm = confusion_matrix(
    actual,
    predicted,
    labels=labels
)

fig, ax = plt.subplots(figsize=(8, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title("Neural Network Round Prediction Confusion Matrix")
plt.xlabel("Predicted Round")
plt.ylabel("Actual Round")

plt.tight_layout()

plt.savefig(
    "results/data_visualizations/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()