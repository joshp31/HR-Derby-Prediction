import subprocess
import sys

scripts = [
    "src/domain_based_models/batted_ball_direction.py",
    "src/domain_based_models/batted_ball_quality.py",
    "src/domain_based_models/batted_ball_type.py",
    "src/domain_based_models/exit_velo.py",
    "src/domain_based_models/launch_angle.py",
    "src/domain_based_models/production_simple.py",
    "src/domain_based_models/production.py",
    "src/domain_based_models/relative_features.py",
]

for script in scripts:

    print(f"Running {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"Completed {script}\n")

print("Pipeline completed successfully.")