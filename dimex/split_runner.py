"""
Module for SPLIT algorithm execution and evaluation.

This module provides:
- train_split: Train a SPLIT model with specified hyperparameters, returning the fitted model,
  its decision tree, and training metadata.
- n_leaves: Count the number of leaf nodes in a binary tree structure.
- prediction_split: Generate predictions with a trained SPLIT model and compute accuracy.
- binarized_features: Extract human-readable names for binarized features from a trained model.
"""

__author__ = "Lucas Campagnaro"

from split import SPLIT
from sklearn.metrics import accuracy_score
import time

def train_split(x_train, y_train,
                lookahead, full_depth, reg,
                verbose=False, binarize=True, time_limit=100):
    """
    Train a SPLIT model with specified hyperparameters and return the fitted model,
    its decision tree, and training metadata.

    Args:
        x_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Labels for training.
        lookahead (int): Lookahead-depth budget for SPLIT.
        full_depth (int): Full-depth budget for SPLIT.
        reg (float): Regularization parameter (λ) for SPLIT.
        verbose (bool, optional): Enable verbose output. Defaults to False.
        binarize (bool, optional): Binarize features before training. Defaults to True.
        time_limit (int, optional): Maximum training time in seconds. Defaults to 100.

    Returns:
        tuple:
            model (SPLIT): The trained SPLIT model instance.
            tree: The underlying decision-tree structure.
            model_data (dict): Metadata including:
                - "lookahead depth": lookahead
                - "full depth": full_depth
                - "lambda": reg
                - "leaves": number of leaves in the tree
                - "features": list of feature names or binarized feature info
                - "runtime": training duration in seconds
    """
    model = SPLIT(lookahead_depth_budget=lookahead, reg=reg, full_depth_budget=full_depth,
                  verbose=verbose, binarize=binarize, time_limit=time_limit)

    # Measure training time
    start = time.perf_counter()
    model.fit(x_train, y_train)
    runtime = time.perf_counter() - start
    
    # Extract tree structure and assemble metadata
    tree = model.tree
    model_data = {
        "lookahead depth": lookahead,
        "full depth": full_depth,
        "lambda": reg,
        "leaves": n_leaves(tree),
        "features": binarized_features(model) if binarize else x_train.columns,
        "runtime": runtime,
    }
    
    return model, tree, model_data

def n_leaves(tree):
    """
    Count the number of leaf nodes in a binary tree structure.

    Args:
        tree: Tree node object with optional attributes 'left_child' and 'right_child'.

    Returns:
        int: Number of leaves (nodes with no children). Returns 0 if tree is None.
    """
    if tree is None:
        return 0
        
    # Retrieve child nodes (None if attribute absent)
    left = getattr(tree, "left_child", None)
    right = getattr(tree, "right_child", None)

    # If no children, it's a leaf
    if left is None and right is None:
        return 1
        
    # Sum leaves from both subtrees
    return n_leaves(left) + n_leaves(right)

def prediction_split(model, x, y):
    """
    Generate predictions with a trained SPLIT model and compute accuracy.

    Args:
        model (SPLIT): Trained SPLIT model instance.
        x (pd.DataFrame): Feature set for prediction.
        y (pd.Series or array-like): True labels for evaluation.

    Returns:
        tuple:
            y_pred (array-like): Predicted labels.
            accuracy (float): Proportion of correct predictions.
    """
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    return y_pred, accuracy

def binarized_features(model):
    """
    Extract human-readable names for binarized features from a trained SPLIT model.

    Args:
        model (SPLIT): Trained SPLIT model instance containing its fitted encoder.

    Returns:
        list of dict: Each dict contains:
            - "index" (int): Position of the encoded feature.
            - "name" (str): Human-readable name of the encoded feature.
    """
    features = []

    # Grab the fitted encoder from the model's internals
    enc = model.__dict__['enc']
    # Retrieve the feature names produced by the encoder
    names = enc.get_feature_names_out()
    
    # Append a dict with index and name for each encoded feature
    for i, name in enumerate(names):
        features.extend([{
            "index": i,
            "name": name,
        }])
        
    return features