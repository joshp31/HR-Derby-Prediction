import subprocess
import sys

scripts = [
    "src/model_2/logistic_regression.py",
    "src/model_2/decision_tree.py",
    "src/model_2/random_forest.py",
    "src/model_2/xg_boost.py",
    "src/model_2/elastic_net.py",
    "src/model_2/knn.py",
    "src/model_2/ordinal_logistic_regression.py",
    "src/model_2/svm.py",
    "src/model_2/neural_network.py",
    "src/model_2/lda.py",
]

for script in scripts:

    print(f"Running {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"Completed {script}\n")

print("Pipeline completed successfully.")