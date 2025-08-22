"""
Module for XGBoost model training, complexity analysis, and evaluation.

This module provides:
- train_xgb: Train an XGBoost classifier and report its size and training time.
- sort_by_gain: Retrieve and sort feature gains from a trained model.
- cumulative_gain: Select top features by gain fraction and retrain the model.
- size: Calculate the number of trees and leaves in a model.
- feature_importance: Plot feature importance using gain as the metric.
- prediction_xgb: Generate predictions and compute accuracy.
"""

__author__ = "Lucas Campagnaro"

import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

def train_xgb(x_train, y_train,
              n_estimators=100, max_depth=3, learning_rate=0.1, 
              objective="binary:logistic", eval_metric="logloss", random_state=42):
    """
    Train an XGBoost classifier and report its size and training time.

    Args:
        x_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series or array-like): Training labels.
        n_estimators (int, optional): Number of boosting rounds. Defaults to 100.
        max_depth (int, optional): Maximum tree depth. Defaults to 3.
        learning_rate (float, optional): Step size shrinkage. Defaults to 0.1.
        objective (str, optional): Learning objective. Defaults to "binary:logistic".
        eval_metric (str, optional): Evaluation metric. Defaults to "logloss".
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple:
            model (xgb.XGBClassifier): Fitted XGBoost model.
            model_size (dict): Counts of trees and leaves in the model.
            runtime (float): Training duration in seconds.
    """
    # Initialize classifier with specified hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_state)

    # Measure training time
    start = time.perf_counter()
    model.fit(x_train, y_train)
    runtime = time.perf_counter() - start

    # Determine model complexity (number of trees and leaves)
    model_size = size(model)

    return model, model_size, runtime

def sort_by_gain(model):
    """
    Retrieve and sort feature gains from an XGBoost model.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model instance.

    Returns:
        tuple:
            gain_sorted (list of tuple): Feature names paired with their gain values, sorted descending.
            total_gain (float): Sum of gain values across all features.
    """
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    
    # Sort features by their gain contribution in descending order
    gain_sorted = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    
    # Aggregate total gain for normalization or thresholding
    total_gain = sum(g for _, g in gain_sorted)
    
    return gain_sorted, total_gain

def cumulative_gain(X_train, y_train,
                    gain_sorted, total_gain, fraction,
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    objective="binary:logistic", eval_metric="logloss", random_state=42):
    """
    Select top features by cumulative gain fraction and retrain an XGBoost model.

    Args:
        X_train (pd.DataFrame): Original training feature matrix.
        y_train (pd.Series): Training labels.
        gain_sorted (list of tuple): Features paired with gain values, sorted descending.
        total_gain (float): Sum of all feature gains.
        fraction (float): Fraction of total gain to retain (between 0 and 1).
        n_estimators (int, optional): Number of boosting rounds. Defaults to 100.
        max_depth (int, optional): Maximum tree depth. Defaults to 3.
        learning_rate (float, optional): Step size shrinkage. Defaults to 0.1.
        objective (str, optional): Learning objective. Defaults to "binary:logistic".
        eval_metric (str, optional): Evaluation metric. Defaults to "logloss".
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple:
            model (xgb.XGBClassifier): Retrained XGBoost model on selected features.
            size (dict): Counts of trees and leaves in the retrained model.
            runtime (float): Training duration in seconds.
            keep (list): Names of features retained.
    """
    cg, keep = 0, []

    # Accumulate gains until the specified fraction of total gain is reached
    for f, g in gain_sorted:
        cg += g
        keep.append(f)
        if cg / total_gain >= fraction:
            break
        
    # Subset training data to the selected features
    X_train_frac = X_train[keep]
    
    # Retrain model on the reduced feature set
    model, size, runtime = train_xgb(X_train_frac, y_train,
                            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            objective=objective, eval_metric=eval_metric, random_state=random_state)

    return model, size, runtime, keep

def size(model):
    """
    Calculate the number of trees and leaves in an XGBoost model.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model instance.

    Returns:
        dict: Counts of "trees" and "leaves" in the model.
    """
    booster = model.get_booster()

    # Extract each tree’s text representation
    trees = booster.get_dump()
    
    # Initialize leaf counter
    leaf_count = 0

    # Total number of trees in the ensemble
    tree_count = len(trees)

    # Count leaf nodes by looking for 'leaf=' in each line of each tree
    for tree in trees:
        for line in tree.splitlines():
            if 'leaf=' in line:
                leaf_count += 1

    size = { "trees": tree_count,
             "leaves": leaf_count,
           }

    return size

def feature_importance(model):
    """
    Plot feature importance for an XGBoost model using the gain metric.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model instance.

    Returns:
        None: Displays a matplotlib plot of feature importance.
    """
    plot_importance(model, importance_type="gain", height=0.4)
    plt.title("Feature importance (gain)")
    plt.tight_layout()
    plt.show()

def prediction_xgb(model, x, y):
    """
    Generate predictions with a trained XGBoost model and compute accuracy.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model instance.
        x (pd.DataFrame or array-like): Feature set for prediction.
        y (pd.Series or array-like): True labels for evaluation.

    Returns:
        tuple:
            y_pred (array-like): Predicted labels.
            accuracy (float): Proportion of correct predictions.
    """
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    return y_pred, accuracy