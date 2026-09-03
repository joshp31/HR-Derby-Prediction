import subprocess
import sys

scripts = [
    "src/data_visualizations/2026_prediction_visualizations.py",
    "src/data_visualizations/accuracy_by_year_visualization.py",
    "src/data_visualizations/confusion_matrix.py",
    "src/data_visualizations/model_accuracy_visualization.py",
    "src/data_visualizations/model_champion_accuracy_visualization.py",
    "src/data_visualizations/predicted_vs_actual_champion_visualization.py"
]

for script in scripts:

    print(f"Running {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"Completed {script}\n")

print("Pipeline completed successfully.")