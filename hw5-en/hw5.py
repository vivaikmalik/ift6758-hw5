import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import shap

# Data Loading and Basic Data Preprocessing


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset as a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data


def encode_target_column(data) -> pd.DataFrame:
    """
    Encode the "is_readmitted" column (target) into numerical values (True -> 1, False -> 0).

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the "is_readmitted" column.

    Returns:
    - pd.DataFrame: DataFrame with the "is_readmitted" column encoded.
    """
    le = LabelEncoder()
    data = data.copy()
    data['is_readmitted'] = le.fit_transform(data['is_readmitted'])

    return data


def split_data(
    data: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """
    Split the data into training and testing sets.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    - tuple: A tuple containing X_train, X_test, y_train, and y_test DataFrames.
    """
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters:
    - X_train (pd.DataFrame): Features of the training set.
    - y_train (pd.Series): Target labels of the training set.
    - n_estimators (int): Number of trees in the forest.
    - max_depth (int): Maximum depth of the trees in the forest (default=None).

    Returns:
    - RandomForestClassifier: Trained Random Forest model.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple:
    """
    Evaluate the Random Forest model.

    Parameters:
    - model (RandomForestClassifier): Trained Random Forest model.
    - X_test (pd.DataFrame): Features of the testing set.
    - y_test (pd.Series): Target labels of the testing set.

    Returns:
    - tuple: A tuple containing accuracy (float) and classification report (str).
    """
    # Complete the function.

    y_pred = model.predict(X_test)
    
    # TODO: Compute the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # TODO: Get the classification report
    report = classification_report(y_test, y_pred)

    results_tuple = (accuracy, report)
    
    return results_tuple


def calculate_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 1,
):
    """
    Calculate permutation importances for a machine learning model.

    Parameters:
    - model: Trained machine learning model.
    - X_val: Validation dataset features.
    - y_val: Validation dataset target labels.

    Returns:
    - eli5.PermutationImportance: PermutationImportance object with calculated importances. We will only use the model and the predefined
    value for the random_state.
    """
    # TODO: Complete the function.

    permutationImportance = PermutationImportance(model, random_state=random_state)
    permutationImportance.fit(X_val, y_val)

    return permutationImportance


def plot_partial_dependence(model, X_val: pd.DataFrame, feature_name: str):
    """
    Display partial dependence plots for a specified feature.

    Parameters:
    - model: Trained machine learning model.
    - X_val: Validation dataset features.
    - feature_name: Name of the feature for which to create partial dependence plots.
    """
    # You can check here the documentation for the required scikit-learn
    # method: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html

    # TODO: Complete the function. Use the name pdp_display for the variable use to store your PD plot object
    # Your code here

    # When you have your code ready uncomment the following code.

    pdp_display = PartialDependenceDisplay.from_estimator(
        model,
        X_val,
        [feature_name]
    )

    pdp_display.figure_.suptitle(f"Partial Dependence Plot for {feature_name}")

    plt.grid(True)


def plot_mean_readmission_vs_time(X_train, y_train):
    """
    Plot the mean readmission rate vs. time in the hospital.

    Parameters:
    - X_train (pd.DataFrame): Features of the training dataset.
    - y_train (pd.Series): Target labels (is_readmitted) of the training dataset.
    """
    # Complete the function.

    # TODO: Combine the features and target labels into a single DataFrame
    all_train = pd.concat([X_train, y_train], axis=1)

    # TODO: Calculate the mean of 'is_readmitted' for each 'time_in_hospital' value

    mean_readmission = all_train.groupby('time_in_hospital')['is_readmitted'].mean() 

    # We will create aninformative and visually appealing plot.

    # No need to modify the following part
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=mean_readmission.index,
        y=mean_readmission.values,
        marker="o",
        color="royalblue",
    )
    plt.xlabel("Time in Hospital")
    plt.ylabel("Mean Readmission Rate")
    plt.title("Mean Readmission Rate vs. Time in Hospital")
    plt.grid(True)

    plt.show()


def main_factors(model: RandomForestClassifier, sample_data: pd.Series):
    """
    Calculate and display SHAP values using a given model and sample data.

    Parameters:
    - model: Trained machine learning model.
    - sample_data: Data for which SHAP values will be calculated and displayed.

    Returns:
    - shap.Explanation: SHAP force plot for the provided data.
    """

    # Complete the function.

    # TODO: Create an object that can calculate SHAP values
    explainer = None

    # TODO: Calculate SHAP values
    shap_values = None

    # TODO: Initialize SHAP JavaScript visualization

    # We create and return a SHAP force plot
    return shap.plots.force(explainer.expected_value[1], shap_values[:, 1], sample_data)


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns_to_process: list,
    predictor_column: str,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove rows with outliers from specific feature columns while ignoring a predictor column using the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing both numeric feature and predictor columns.
        columns_to_process (list): A list of column names to process for outlier removal.
        predictor_column (str): The name of the predictor column to be ignored during outlier detection.
        threshold (float, optional): The threshold multiplier for defining outlier bounds. Default is 1.5.

    Returns:
        pd.DataFrame: A cleaned DataFrame with outlier rows removed.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = df.copy()

    # TODO: Complete the function.

    # TODO: Loop through each specified feature column. Uncomment the for loop and the if statement when you are ready to test your code.
    def detect_outliers(df, feature):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        return outliers.index
        
    outlier_indices = set()
    
    for column in columns_to_process:
        if column != predictor_column and column in df.columns:
    # TODO: Carry out the necessary steps to implement the IQR Method.
            df_cleaned[column] = df_cleaned[column].astype(float)
            outlier_idx = detect_outliers(df_cleaned, column)
            outlier_indices.update(outlier_idx)  # add indices of outliers
            

    # TODO: Identify and remove rows with values outside the bounds
    df_cleaned = df_cleaned.drop(index=outlier_indices)
    # TODO: Reset the index of the clean dataframe
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned


def add_absolute_coordinate_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns 'abs_lon_change' and 'abs_lat_change' to an existing DataFrame, representing the absolute change
    in longitude and latitude between 'dropoff' and 'pickup' coordinates.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', and 'dropoff_latitude' columns.

    Returns:
        pd.DataFrame: The DataFrame with the added columns.
    """
    df = df.copy()

    df['abs_lon_change'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['abs_lat_change'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])
    # TODO: Calculate absolute changes in longitude and latitude

    return df
