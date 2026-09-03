MLB Home Run Derby Prediction

A machine learning project that uses historical MLB batting and Statcast data to predict how far Home Run Derby participants will advance in the modern four-round Home Run Derby format.

Project Overview

The goal of this project is to predict Home Run Derby outcomes using machine learning. Since the current Home Run Derby format was introduced in 2015, participants have competed through a bracket consisting of four possible outcomes: Round 1, Round 2, Round 3, or Champion.

Rather than predicting a continuous statistic such as the number of home runs hit, this project treats the Derby as a four-class classification problem. Each participant is assigned a Round_Reached value from 1–4:

Round	Outcome
1	Eliminated in the first round
2	Advances to the semifinals
3	Advances to the final
4	Wins the Home Run Derby

Predicting the Home Run Derby presents a unique machine learning problem because the Derby is fundamentally different from a traditional baseball game. Traditional offensive statistics provide useful information about a player's power and production, but Home Run Derby performance depends on skills and characteristics that are not necessarily captured by conventional statistics.

Another challenge is that the strength of the field varies from year to year. A player who ranks highly within one Derby field may rank much lower in another. Therefore, the model must consider not only a player's absolute performance, but also how that player's statistics compare with the other participants in the same Derby.

This project addresses these challenges by combining historical baseball statistics, Statcast metrics, engineered performance features, player-relative features, multiple machine learning approaches, cross-validation, and a probability-based method for assigning players to valid Derby outcomes.

Objectives

The primary objectives of this project were to:

Engineer meaningful baseball and Statcast features for Home Run Derby prediction
Compare multiple machine learning classification methods
Evaluate models using cross-validation
Identify the model with the strongest overall and champion prediction performance
Determine which statistical characteristics provide useful predictive information
Build a reproducible data engineering pipeline using a medallion architecture
Generate a true out-of-sample prediction for the 2026 Home Run Derby
Visualize model predictions, probabilities, and outcomes

Data

Data Sources

Player statistics were obtained using the pybaseball Python library, which provides access to MLB and Baseball Reference data.

Historical Home Run Derby participants and results were compiled from publicly available sources, including Wikipedia's 2026 Major League Baseball Home Run Derby page.

Time Period

The dataset covers Home Run Derby participants from 2015 through 2026. The 2020 Home Run Derby was excluded because the event was not held due to the COVID-19 pandemic.

Player statistics represent performance accumulated during the season through the All-Star break, ensuring that the model only uses information that would have been available at the time of the Home Run Derby.

Target Variable

The target variable is:

Round_Reached

with four possible outcomes:

Value	Outcome
1	Round 1
2	Round 2
3	Round 3
4	Champion

Features

The dataset includes traditional batting statistics, Statcast metrics, batted-ball characteristics, and engineered relative statistics.

Examples include:

Plate appearances
Home runs
OPS
ISO
At-bats per home run
Walk percentage
Strikeout percentage
Extra-base hit percentage
Total bases per plate appearance
Average exit velocity
Maximum exit velocity
95th-percentile exit velocity
Average launch angle
Median launch angle
Launch-angle standard deviation
Hard-hit percentage
Barrel percentage
Sweet-spot percentage
Air-ball percentage
Ground-ball percentage
Popup percentage
Pull percentage
Center-field percentage

Relative features were also created to measure how a player's performance compared with the other players in the same Home Run Derby field.

Data Engineering

The project uses a medallion architecture to organize the data engineering process into Bronze, Silver, and Gold layers.

Bronze

The Bronze layer contains raw or minimally processed data ingested from the underlying data sources.

This layer is responsible for collecting historical player statistics and preserving the source data before significant transformations are applied.

Silver

The Silver layer cleans and filters the raw data and narrows it to the players relevant to the Home Run Derby.

This stage includes:

Identifying Home Run Derby participants
Matching players to MLB identifiers
Filtering statistics to the appropriate season
Restricting statistics to information available before the Derby
Combining participant and statistical datasets

Gold

The Gold layer contains the final datasets used for modeling.

These datasets contain:

Cleaned player statistics
Engineered baseball metrics
Player-relative features
The target variable
Model-ready observations

This architecture separates data ingestion, transformation, and modeling, making the project easier to reproduce and modify.

Feature Engineering

Feature engineering was used to transform raw baseball statistics into variables that better represent characteristics relevant to Home Run Derby performance.

Rate and Percentage Features

Several statistics were converted into rates or percentages to make player performance more comparable.

Examples include:

At-bats per home run
Walk percentage
Strikeout percentage
Extra-base hit percentage
Total bases per plate appearance
Hard-hit percentage
Barrel percentage
Sweet-spot percentage

Statcast Features

Statcast measurements were incorporated to capture the quality and characteristics of a player's contact.

Examples include:

Average exit velocity
Maximum exit velocity
95th-percentile exit velocity
Average launch angle
Median launch angle
Launch-angle standard deviation
Relative Features

A major component of the feature engineering process was creating features relative to the other participants in the same Derby.

For example:

EV_95_above_avg

measures how far a player's 95th-percentile exit velocity is above the average of the relevant Home Run Derby field.

This approach accounts for differences in the strength of each year's field. Instead of asking only whether a player has a high exit velocity, the model can evaluate whether the player has a high exit velocity relative to the competition they face that year.

Modeling Methodology

The problem was formulated as a four-class classification problem with Round_Reached as the target variable.

Multiple machine learning methods were evaluated, including:

Logistic Regression
K-Nearest Neighbors
Decision Tree
Random Forest
XGBoost
Elastic Net
Ordinal Logistic Regression
Support Vector Machine
Neural Network
Linear Discriminant Analysis

Models were evaluated based on their ability to correctly predict the round reached by Home Run Derby participants.

Two primary performance measures were tracked.

Overall Accuracy

The percentage of participants for whom the model correctly predicted the player's final round.

Champion Accuracy

The percentage of Home Run Derby fields for which the model correctly identified the eventual champion.

Champion accuracy was tracked separately because correctly identifying the winner is a particularly meaningful outcome for a Home Run Derby prediction model.

Cross-Validation

Cross-validation was used during model evaluation and feature selection to estimate how well each model would generalize to unseen Home Run Derby participants.

Because multiple participants come from the same Derby field, observations from the same year are not completely independent. To account for this structure, the project used Leave-One-Group-Out (LOGO) cross-validation, with the Derby year serving as the grouping variable.

For each validation iteration:

All participants from one Derby year were held out as the validation set.
The model was trained using participants from all other Derby years.
Predictions were generated for the held-out Derby field.
Accuracy was calculated from the held-out predictions.
The process was repeated until every Derby year had served as the validation group.

This approach tests whether a model can generalize from previous Derby fields to a completely unseen field rather than simply predicting additional players from years it has already seen.

Cross-validation was used to compare models and feature configurations before selecting the final model.

The 2026 Home Run Derby was not included in model selection or cross-validation for the final prediction. Instead, 2026 was reserved as a true out-of-sample evaluation.

Model Selection

After comparing the evaluated models and feature combinations, the final model selected for the 2026 prediction was a Neural Network.

Final Features

The final model used three features:

Max_EV
EV_95_above_avg
sweet_spot%

These features capture different aspects of a player's ability to generate high-quality contact:

Max_EV: Maximum observed exit velocity
EV_95_above_avg: 95th-percentile exit velocity relative to the average of the Derby field
sweet_spot%: Percentage of batted balls within the optimal launch-angle range

Final Model

The final neural network was implemented using MLPClassifier with the following configuration:

Hidden Layers: (5, 5)
Alpha: 0.1
Maximum Iterations: 5000
Random State: 31

The final pipeline also standardized the input features using StandardScaler.

Model Performance

The final Neural Network achieved:

Metric	Performance
Overall Accuracy	56.25%
Champion Accuracy	50.00%

The Neural Network was selected as the final model based on its cross-validation performance among the evaluated models and feature configurations.

2026 Out-of-Sample Prediction

The 2026 Home Run Derby was treated as a true out-of-sample test.

After selecting the final model and feature set, the model was retrained using all available pre-2026 data. The model was then used once to generate predictions for the 2026 Home Run Derby.

No 2026 Derby outcome information was used during model selection or training, making 2026 a true out-of-sample evaluation.

Bracket Assignment

Because the Home Run Derby has a fixed bracket structure, predictions cannot be treated as completely independent classifications.

The model first generates probabilities for each player across all four possible outcomes:

Round 1
Round 2
Round 3
Champion

The resulting probabilities are then used to assign the eight participants to a valid Derby structure:

4 players → Round 1
2 players → Round 2
1 player → Round 3
1 player → Champion

The assignment procedure evaluates all valid combinations of round assignments and selects the configuration that maximizes the overall log probability of the predicted outcomes.

This ensures that the final predictions respect the actual structure of the Home Run Derby.

Results and Visualizations

Detailed model results and visualizations can be found in the results folder.

The folder contains individual results for each model evaluated, along with summary visualizations of model performance and the final 2026 predictions. These files provide additional detail on the modeling process and outcomes beyond what is summarized in this README.

Key Findings
1. High-End Exit Velocity Was Important

Maximum exit velocity and high-end exit velocity relative to the competition were among the final model's selected features. The final model therefore placed substantial predictive value on a player's ability to generate high-quality contact.

2. Relative Performance Provides Additional Context

Comparing players to the other participants in their Derby field can occasionally provide information that absolute statistics alone cannot capture.

A player's statistics can have different implications depending on the strength of the field they enter.

3. Home Run Derby Performance Is Difficult to Predict

The Home Run Derby contains substantial variability that traditional season-long statistics cannot completely explain. While player power metrics provide useful predictive information, the event itself is highly specialized and contains relatively few observations per season.

4. Probability-Based Predictions Provide More Information

Rather than treating every prediction as equally certain, the model produces probabilities for all four possible outcomes. This allows the model's confidence and uncertainty to be examined in addition to its final classification.

5. 2026 Provides a True Out-of-Sample Evaluation

The final model was trained exclusively on information from before 2026 and was then evaluated against the actual 2026 Derby results. This provides a more realistic assessment of how the model performs on a future Derby field.

Technologies Used
Programming
Python
Data Engineering & Analysis
Pandas
NumPy
PyBaseball
Machine Learning
Scikit-learn
Statsmodels
XGBoost
Visualization
Matplotlib
Seaborn
Development
Git
GitHub
VS Code

Future Improvements

Potential future improvements include:

Explore environmental factors such as ballpark dimensions and weather
Expand the historical dataset as additional Home Run Derby events occur
Investigate additional ordinal classification techniques
Explore more advanced neural network architectures
Incorporate additional contextual information about each Derby field
Incorporate additional Statcast and player-level features
Explore additional methods for modeling the structure of the Home Run Derby bracket

Author

Josh Purvis

https://github.com/joshp31 | https://www.linkedin.com/in/purvisjosh/