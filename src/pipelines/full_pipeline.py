import subprocess
import sys

scripts = [
    "src/data_engineering/Bronze_to_Silver_HR_Derby_Participants.py",
    "src/data_engineering/Ingest_Bronze_Statcast_Batting.py",
    "src/data_engineering/Bronze_to_Silver_Statcast_Batting.py",
    "src/data_engineering/Gold_Derby_Player.py",
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
    "src/data_engineering/Gold_Derby_Player_W_Relative_Features.py",
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
    "src/domain_based_models/batted_ball_direction.py",
    "src/domain_based_models/batted_ball_quality.py",
    "src/domain_based_models/batted_ball_type.py",
    "src/domain_based_models/exit_velo.py",
    "src/domain_based_models/launch_angle.py",
    "src/domain_based_models/production_simple.py",
    "src/domain_based_models/production.py",
    "src/domain_based_models/relative_features.py",
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
    "src/domain_based_models_combined/production_simple_+_launch_angle.py",
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