"""Helper functions for the NBA MVP prediction project."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_linear_regression(complete_data: pd.DataFrame, cutoff_year: int, features: list[str], target: str) -> tuple:
    """Generate a linear regression model and return the pipeline, OLS, X_test, y_test, ids_test, and X_train.

    Args:
        complete_data (pd.DataFrame): Dataframe containing all the data.
        cutoff_year (int): Cutoff between training and testing data.
        features (list[str]): Features to use in the model.
        target (str): Target variable to predict.

    Returns:
        tuple: A tuple containing the pipeline, OLS, X_test, y_test, ids_test, and X_train.
    """
    # Splitting complete_data while keeping mvp_vote_share as the target variable
    train_data = complete_data[complete_data['season'] <= cutoff_year]
    test_data = complete_data[complete_data['season'] > cutoff_year]

    # Now, split the train_data and test_data into features and identifiers
    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    ids_test = test_data[["seas_id", "season", "player_id", "player", "tm", "mvp_race_rank"]]
    y_test = test_data[target]
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(X_train, y_train)
    OLS = pipeline.named_steps['linearregression']
    return pipeline, OLS, X_test, y_test, ids_test, X_train


def optimize_threshold(y_test: pd.Series, y_pred: np.ndarray, num_iterations: int, step_size: float) -> tuple:
    """Find the best threshold to set low predictions to zero.

    Args:
        y_test (pd.Series): The actual values of the target variable.
        y_pred (np.ndarray): The predicted values of the target variable.
        num_iterations (int): The number of iterations to try.
        step_size (float): The step size to use in the iterations.

    Returns:
        tuple: Lists of thresholds and R-squared scores, the best threshold, and the best R-squared score.
    """
    thresholds = np.arange(0, num_iterations * step_size, step_size)
    r2_scores = []

    for current_threshold in thresholds:
        # Post-process predictions
        y_pred_processed = np.where(y_pred < current_threshold, 0, y_pred)
        # Calculate R squared score
        current_r2 = r2_score(y_test, y_pred_processed)
        r2_scores.append(current_r2)

    # Find the index of the best R-squared score
    best_idx = np.argmax(r2_scores)
    best_threshold = thresholds[best_idx]
    best_r2 = r2_scores[best_idx]

    return thresholds, r2_scores, best_threshold, best_r2


def graph_key_features(OLS: LinearRegression, X_train: pd.DataFrame) -> None:
    """Graph the key features of the model.

    Args:
        OLS (LinearRegression): The linear regression model.
        X_train (pd.DataFrame): The training data.
    """
    coefficients = OLS.coef_

    features = X_train.columns.tolist()

    # Pair each feature with its coefficient
    feature_importance = zip(features, coefficients)

    # Sort features by absolute value of their coefficient, in descending order
    sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

    features, coefficients = zip(*sorted_features)

    # Create the bar graph
    plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed
    plt.barh(features, coefficients)  # Using horizontal bar chart for better readability
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Feature Coefficients')
    plt.gca().invert_yaxis()  # Invert y-axis to have the largest coefficient on top
    plt.show()


def get_highest_estimated_mvp_vote_share_seasons(X_test: pd.DataFrame, y_pred: np.ndarray, ids_test: pd.DataFrame) -> None:
    """Get the top 10 highest predictions and their corresponding features.

    Args:
        X_test (pd.DataFrame): The test data.
        y_pred (np.ndarray): The predictions.
        ids_test (pd.DataFrame): The identifiers corresponding to the test data.
    """
    X_test_with_predictions = X_test.copy()
    X_test_with_predictions['Predicted'] = y_pred
    # Add identifiers to the DataFrame
    X_test_with_predictions = pd.concat([ids_test, X_test_with_predictions], axis=1)

    # Sort by the 'Predicted' column to get the top 10 highest predictions
    top_10_predictions = X_test_with_predictions.sort_values(by='Predicted', ascending=False).head(10)

    print("Top 10 Predictions and Corresponding Features:")
    print(top_10_predictions[['player', 'Predicted', 'season']])


def get_mvp_for_test_seasons(X_test: pd.DataFrame, y_pred: np.ndarray, ids_test: pd.DataFrame) -> None:
    """Get the predicted MVP for each season in the test data.

    Args:
        X_test (pd.DataFrame): The test data.
        y_pred (np.ndarray): The predictions.
        ids_test (pd.DataFrame): The identifiers corresponding to the test data.
    """
    # Group by 'seas_id' and find the index of the max 'mvp_vote_share' in each group
    X_test_with_predictions = X_test.copy()
    X_test_with_predictions['Predicted'] = y_pred
    # Add identifiers to the DataFrame
    X_test_with_predictions = pd.concat([ids_test, X_test_with_predictions], axis=1)

    idx = X_test_with_predictions.groupby('season')['Predicted'].idxmax()

    # Use the index to select the rows
    highest_votes_per_season = X_test_with_predictions.loc[idx]

    print(highest_votes_per_season[['player', 'Predicted', 'season']])


def calculate_points(predicted_rank):
    # Award points based on the predicted rank of the actual MVP
    if predicted_rank == 1:
        return 10
    elif predicted_rank == 2:
        return 5
    elif predicted_rank == 3:
        return 3
    elif predicted_rank == 4:
        return 2
    elif predicted_rank == 5:
        return 1
    else:
        return 0


def get_mvp_prediction_accuracy(predicted_mvp_df: pd.DataFrame, actual_mvp_df: pd.DataFrame) -> float:
    """Generate accuracy score for MVP predictions.

    Args:
        predicted_mvp_df (pd.DataFrame): Dataframe containing predicted MVPs.
        actual_mvp_df (pd.DataFrame): Dataframe containing actual MVPs.

    Returns:
        float: Score representing the accuracy of the MVP predictions.
    """
    # Filter actual MVPs (rank 1)
    actual_mvps = actual_mvp_df[actual_mvp_df['mvp_race_rank'] == 1]

    # Merge the predicted rankings with the actual MVPs on player and season
    merged_df = pd.merge(actual_mvps, predicted_mvp_df, on=['player', 'season'])

    # Calculate points for how close each prediction came to actual MVP
    merged_df['points'] = merged_df['predicted_mvp_race_rank'].apply(calculate_points)

    # Calculate the average score
    average_score = merged_df['points'].mean()
    return average_score


# Mapping of team names to conferences (Not 100 percent accurate, but close enough for our purposes)
conference_mapping = {
    'Atlanta Hawks': 'Eastern',
    'Boston Celtics': 'Eastern',
    'Brooklyn Nets': 'Eastern',
    'Buffalo Braves': 'Eastern',
    'New Jersey Nets': 'Eastern',
    'Chicago Bulls': 'Eastern',
    'Charlotte Hornets': 'Eastern',
    'Charlotte Bobcats': 'Eastern',
    'Cleveland Cavaliers': 'Eastern',
    'Dallas Mavericks': 'Western',
    'Denver Nuggets': 'Western',
    'Detroit Pistons': 'Eastern',
    'Golden State Warriors': 'Western',
    'Houston Rockets': 'Western',
    'Indiana Pacers': 'Eastern',
    'Los Angeles Clippers': 'Western',
    'San Diego Clippers': 'Western',
    'Los Angeles Lakers': 'Western',
    'Memphis Grizzlies': 'Western',
    'Vancouver Grizzlies': 'Western',
    'Miami Heat': 'Eastern',
    'Milwaukee Bucks': 'Eastern',
    'Minnesota Timberwolves': 'Western',
    'New Orleans Pelicans': 'Western',
    'New Orleans Jazz': 'Western',
    'New Orleans Hornets': 'Western',
    'New Orleans/Oklahoma City Hornets': 'Western',
    'New York Knicks': 'Eastern',
    'New York Nets': 'Eastern',
    'Oklahoma City Thunder': 'Western',
    'Seattle SuperSonics': 'Western',
    'Orlando Magic': 'Eastern',
    'Philadelphia 76ers': 'Eastern',
    'Phoenix Suns': 'Western',
    'Portland Trail Blazers': 'Western',
    'Sacramento Kings': 'Western',
    'Kansas City Kings': 'Western',
    'San Antonio Spurs': 'Western',
    'Toronto Raptors': 'Eastern',
    'Utah Jazz': 'Western',
    'Washington Wizards': 'Eastern',
    'Washington Bullets': 'Eastern',
    'Buffalo Braves': 'Eastern',
    'New York Nets': 'Eastern'
}
