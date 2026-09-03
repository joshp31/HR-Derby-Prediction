import subprocess
import sys

scripts = [
    "src/domain_based_models_combined/all.py",
    "src/domain_based_models_combined/batted_ball_direction_+_quality_+_type.py",
    "src/domain_based_models_combined/batted_ball_direction_+_quality.py",
    "src/domain_based_models_combined/exit_velo_+_batted_ball_direction_+_quality.py",
    "src/domain_based_models_combined/exit_velo_+_batted_ball_direction.py",
    "src/domain_based_models_combined/exit_velo_+_batted_ball_quality_+_type.py",
    "src/domain_based_models_combined/exit_velo_+_batted_ball_quality.py",
    "src/domain_based_models_combined/exit_velo_+_batted_ball_type.py",
    "src/domain_based_models_combined/exit_velo_+_launch_angle.py",
    "src/domain_based_models_combined/forward_sequential_selection_result.py",
    # "src/domain_based_models_combined/forward_sequential_selection.py", (Long runtime, chooses model used in forward_sequential_selection_result.py)
    "src/domain_based_models_combined/intuitive.py",
    "src/domain_based_models_combined/optimal.py",
    "src/domain_based_models_combined/production_simple_+_batted_ball_direction.py",
    "src/domain_based_models_combined/production_simple_+_batted_ball_quality_+_type.py",
    "src/domain_based_models_combined/production_simple_+_batted_ball_quality.py",
    "src/domain_based_models_combined/production_simple_+_batted_ball_type.py",
    "src/domain_based_models_combined/production_simple_+_exit_velo.py",
    "src/domain_based_models_combined/production_simple_+_launch_angle.py"
]

for script in scripts:

    print(f"Running {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"Completed {script}\n")

print("Pipeline completed successfully.")