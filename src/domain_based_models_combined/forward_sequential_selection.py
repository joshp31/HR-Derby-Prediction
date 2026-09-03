import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "data/gold/statcast_batting_w_relative_features/batter_stats_relative_pre_2026.csv"
)

features = [
    "plate_appearances",
    
    "home_runs",
    
    "Avg_EV",
    "Max_EV",
    "EV_95",
    "Avg_LA",
    "Median_LA",
    "Std_LA",
    
    "ops",
    "iso",
    "ab/hr",
    "bb%",
    "k%",
    "xbh%",
    "total_bases_per_pa",
    
    "hard_hit%",
    "barrel%",
    "sweet_spot%",
    
    "air_ball%",
    "ground_ball%",
    "popup%",
    
    "pull_percentage",
    "center_percentage",

    "plate_appearances_above_avg",

    "home_runs_above_avg",

    "Avg_EV_above_avg",
    "Max_EV_above_avg",
    "EV_95_above_avg",
    "Avg_LA_above_avg",
    "Median_LA_above_avg",
    "Std_LA_above_avg",

    "ops_above_avg",
    "iso_above_avg",
    "ab/hr_above_avg",
    "bb%_above_avg",
    "k%_above_avg",
    "xbh%_above_avg",
    "total_bases_per_pa_above_avg",

    "hard_hit%_above_avg",
    "barrel%_above_avg",
    "sweet_spot%_above_avg",

    "air_ball%_above_avg",
    "ground_ball%_above_avg",
    "popup%_above_avg",

    "pull_percentage_above_avg",
    "center_percentage_above_avg"
]

X = df[features]
y = df["Round_Reached"]
groups = df["Year"]

model = Pipeline([
    ("scaler", StandardScaler()),

    ("neural_network", MLPClassifier(
        alpha=0.1,
        hidden_layer_sizes=(5,5),
        max_iter=5000,
        random_state=31
    ))
])

def assign_rounds(probabilities):

    n_players = len(probabilities)

    if n_players != 8:
        raise ValueError(
            f"Expected 8 players, but received {n_players}"
        )

    best_score = -np.inf
    best_assignment = None

    players = range(n_players)

    for round_4 in itertools.combinations(players, 1):

        remaining_after_4 = [
            p for p in players
            if p not in round_4
        ]

        for round_3 in itertools.combinations(
            remaining_after_4, 1
        ):

            remaining_after_3 = [
                p for p in remaining_after_4
                if p not in round_3
            ]

            for round_2 in itertools.combinations(
                remaining_after_3, 2
            ):

                round_1 = [
                    p for p in remaining_after_3
                    if p not in round_2
                ]

                assignment = {}

                for p in round_1:
                    assignment[p] = 1

                for p in round_2:
                    assignment[p] = 2

                for p in round_3:
                    assignment[p] = 3

                for p in round_4:
                    assignment[p] = 4

                score = 0

                for p in players:

                    round_number = assignment[p]

                    probability = probabilities[
                        p,
                        round_number - 1
                    ]

                    probability = max(
                        probability,
                        1e-15
                    )

                    score += np.log(probability)

                if score > best_score:

                    best_score = score
                    best_assignment = assignment

    return np.array([
        best_assignment[p]
        for p in players
    ])

def sequential_feature_selection(
    X,
    y,
    groups
):

    selected_features = []
    remaining_features = list(X.columns)

    best_accuracy = -np.inf

    while len(remaining_features) > 0:

        feature_results = []

        for feature in remaining_features:

            candidate_features = (
                selected_features + [feature]
            )

            candidate_X = X[candidate_features]

            logo = LeaveOneGroupOut()

            predictions = []
            actual = []

            for train_idx, test_idx in logo.split(
                candidate_X,
                y,
                groups=groups
            ):

                X_train = candidate_X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                X_test = candidate_X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                candidate_model = Pipeline([
                    ("scaler", StandardScaler()),

                    ("neural_network", MLPClassifier(
                        alpha=0.1,
                        hidden_layer_sizes=(5,5),
                        max_iter=5000,
                        random_state=31
                    ))
                ])

                candidate_model.fit(
                    X_train,
                    y_train
                )

                probabilities = (
                    candidate_model.predict_proba(
                        X_test
                    )
                )

                fold_predictions = assign_rounds(
                    probabilities
                )

                predictions.extend(
                    fold_predictions
                )

                actual.extend(
                    y_test
                )

            accuracy = accuracy_score(
                actual,
                predictions
            )

            feature_results.append(
                (
                    feature,
                    accuracy
                )
            )

        feature, feature_accuracy = max(
            feature_results,
            key=lambda x: x[1]
        )

        if feature_accuracy > best_accuracy:

            selected_features.append(feature)

            remaining_features.remove(feature)

            best_accuracy = feature_accuracy

            print(
                f"Added: {feature} | "
                f"Accuracy: {feature_accuracy:.4f}"
            )

        else:

            break

    return selected_features

selected_features = sequential_feature_selection(
    X,
    y,
    groups
)

print("\nSelected Features:")
for feature in selected_features:
    print(feature)

X = df[selected_features]

logo = LeaveOneGroupOut()

year_results = []

all_predictions = []
all_actual = []

for train_idx, test_idx in logo.split(
    X,
    y,
    groups=groups
):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    year = groups.iloc[test_idx].iloc[0]

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(X_test)

    predictions = assign_rounds(
        probabilities
    )

    for i, idx in enumerate(test_idx):

        row = df.iloc[idx].copy()

        row["Predicted_Round"] = predictions[i]

        for class_idx, class_value in enumerate(
            model.classes_
        ):

            row[
                f"Probability_Round_{class_value}"
            ] = probabilities[
                i,
                class_idx
            ]

        row["Correct"] = (
            predictions[i]
            == y_test.iloc[i]
        )

        year_results.append(row)

    all_predictions.extend(
        predictions
    )

    all_actual.extend(
        y_test
    )

results_df = pd.DataFrame(
    year_results
)

overall_accuracy = accuracy_score(
    all_actual,
    all_predictions
)

for year in sorted(
    results_df["Year"].unique()
):

    year_df = results_df[
        results_df["Year"] == year
    ]

    accuracy = (
        year_df["Predicted_Round"]
        == year_df["Round_Reached"]
    ).mean()

for year in sorted(
    results_df["Year"].unique()
):

    distribution = (
        results_df[
            results_df["Year"] == year
        ]["Predicted_Round"]
        .value_counts()
        .sort_index()
    )

for year in sorted(
    results_df["Year"].unique()
):

    year_df = results_df[
        results_df["Year"] == year
    ]

    actual_champion = year_df[
        year_df["Round_Reached"] == 4
    ]

    predicted_champion = year_df[
        year_df["Predicted_Round"] == 4
    ]

    actual_name = (
        actual_champion["player_name"]
        .iloc[0]
    )

    predicted_name = (
        predicted_champion["player_name"]
        .iloc[0]
    )

    champion_correct = (
        actual_name == predicted_name
    )

results_path = (
    "results/domain_based_models_combined/forward_sequential_selection_results.txt"
)

with open(results_path, "w") as f:

    f.write("Neural Network Results\n")
    f.write("======================\n\n")

    f.write(
        f"Overall Accuracy: {overall_accuracy:.4f}\n"
    )

    f.write("\n\n")

    f.write("Selected Features\n")
    f.write("-----------------\n")

    for feature in selected_features:
        f.write(
            f"{feature}\n"
        )

    f.write("\n\n")

    f.write("Accuracy by Year\n")
    f.write("----------------\n")

    accuracy_by_year = []

    for year in sorted(
        results_df["Year"].unique()
    ):
        year_df = results_df[
            results_df["Year"] == year
        ]

        accuracy = (
            year_df["Predicted_Round"]
            == year_df["Round_Reached"]
        ).mean()

        accuracy_by_year.append({
            "year": year,
            "accuracy": accuracy
        })

    f.write("\n\n")

    f.write("Champion Predictions\n")
    f.write("--------------------\n")

    champion_correct_count = 0

    total_years = len(
        results_df["Year"].unique()
    )

    champion_results = []

    for year in sorted(
        results_df["Year"].unique()
    ):

        year_df = results_df[
            results_df["Year"] == year
        ]

        actual_champion = year_df[
            year_df["Round_Reached"] == 4
        ]

        predicted_champion = year_df[
            year_df["Predicted_Round"] == 4
        ]

        actual_name = (
            actual_champion["player_name"]
            .iloc[0]
        )

        predicted_name = (
            predicted_champion["player_name"]
            .iloc[0]
        )

        champion_correct = (
            actual_name == predicted_name
        )

        champion_results.append({
            "Year": year,
            "Actual_Champion": actual_name,
            "Predicted_Champion": predicted_name,
            "Correct": champion_correct
        })

        if champion_correct:
            champion_correct_count += 1

        f.write(
            f"{year}: "
            f"Actual = {actual_name} | "
            f"Predicted = {predicted_name} | "
            f"Correct = {champion_correct}\n"
        )

    champion_accuracy = (
        champion_correct_count / total_years
    )

    f.write("\n")

    f.write(
        f"Champion Accuracy: "
        f"{champion_accuracy:.4f}\n"
    )

prediction_results = pd.DataFrame({
    "actual_round": all_actual,
    "predicted_round": all_predictions
})

prediction_results.to_csv(
    "results/model_predictions.csv",
    index=False
)

overall_results_path = "results/overall_model_results.csv"

columns = [
    "model",
    "accuracy",
    "champion_accuracy"
]

new_result = pd.DataFrame([{
    "model": "Neural Network - Forward Sequential Selection",
    "accuracy": overall_accuracy,
    "champion_accuracy": champion_accuracy
}])

try:
    overall_results = pd.read_csv(overall_results_path)

    if overall_results.empty:
        overall_results = pd.DataFrame(columns=columns)

except (FileNotFoundError, pd.errors.EmptyDataError):
    overall_results = pd.DataFrame(columns=columns)

overall_results = overall_results[
    overall_results["model"] != "Neural Network - Forward Sequential Selection"
]

if overall_results.empty:
    overall_results = new_result
else:
    overall_results = pd.concat(
        [overall_results, new_result],
        ignore_index=True
    )

overall_results.to_csv(
    overall_results_path,
    index=False
)

accuracy_by_year_df = pd.DataFrame(
    accuracy_by_year
)

accuracy_by_year_df.to_csv(
    "results/accuracy_by_year.csv",
    index=False
)

champion_results_df = pd.DataFrame(
    champion_results
)

champion_results_df.to_csv(
    "results/champion_predictions.csv",
    index=False
)