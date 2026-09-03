import subprocess
import sys

scripts = [
    "src/model_1/logistic_regression.py",
    "src/model_1/decision_tree.py",
    "src/model_1/random_forest.py",
    "src/model_1/xg_boost.py",
    "src/model_1/elastic_net.py",
    "src/model_1/knn.py",
    "src/model_1/ordinal_logistic_regression.py",
    "src/model_1/svm.py",
    "src/model_1/neural_network.py",
    "src/model_1/lda.py",
]

for script in scripts:

    print(f"Running {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"Completed {script}\n")

print("Pipeline completed successfully.")