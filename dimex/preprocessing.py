"""
Preprocessing utilities for dataset preparation and encoding.

This module provides functions to:
- Remove missing data and report cleaning statistics.
- Encode categorical variables into one-hot numeric formats.
- Binarize target labels for binary classification.
- Balance class distributions via mid-point undersampling or SMOTE.
- Split datasets into stratified training and test sets.
- Compute basic class distribution statistics.

Each function returns processed data structures and, where applicable, saves the result to a new CSV file.
"""

__author__ = "Lucas Campagnaro"

import pandas as pd
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os
import re

def midrange_undersample(csv_path, index_col=0, random_state=42, index=False, drop=True):
    """
    Undersample a CSV dataset to a 50/50 split around its label mid-point.

    Args:
        csv_path (str): Path to the CSV file.
        index_col (int, optional): Column to use as index. Defaults to 0.
        random_state (int, optional): Seed for random sampling. Defaults to 42.
        index (bool, optional): Whether to write index when saving. Defaults to False.
        drop (bool, optional): If True, drop original index when resetting. Defaults to True.

    Returns:
        pandas.DataFrame: Original DataFrame if already balanced; otherwise, the new balanced DataFrame.

    Raises:
        UserWarning: If the minority/majority ratio falls below the skew threshold.

    Examples:
        >>> df = midrange_undersample("data.csv")
        >>> balance_stats(df.iloc[:, -1])
        {'1': 0.5, '0': 0.5}
    """
    # If one class is ≥3× larger than the other, balancing drops >50% of samples
    SKEW_THRESHOLD = 0.33
    
    dataset = pd.read_csv(csv_path, index_col=index_col)

    labels = dataset.columns[-1] # name of the labels column

    # Compute mid-point between min and max label values
    min_value = dataset[labels].min()
    max_value = dataset[labels].max()
    mid = (min_value + max_value) / 2
    
    # Partition samples above, below, and exactly at mid-point
    dataset_true = dataset[dataset[labels] > mid]
    dataset_false = dataset[dataset[labels] < mid]
    dataset_mid = dataset[dataset[labels] == mid]

    
    if len(dataset_true) == len(dataset_false):
        return dataset
    else:
        # Determine majority vs. minority
        if len(dataset_true) > len(dataset_false): # if True labels > False labels
            # Warn if data is too skewed to balance without heavy loss
            ratio = len(dataset_false) / len(dataset_true)
            if ratio < SKEW_THRESHOLD:
                warnings.warn(f'Your data is extremely skewed (minority / majority = {ratio:.2f}). Balancing will remove too much data.')
            # Sample majority class down to minority size
            dataset_true_sampled = dataset_true.sample(n=len(dataset_false), random_state=random_state)
            dataset_balanced = pd.concat([dataset_true_sampled, dataset_false, dataset_mid]).sample(frac=1, random_state=random_state).reset_index(drop=drop)
        else:
            ratio = len(dataset_true) / len(dataset_false)
            if ratio <= SKEW_THRESHOLD:
                warnings.warn(f'Your data is extremely skewed (minority / majority = {ratio:.2f}). Balancing will remove too much data.')   
            dataset_false_sampled = dataset_false.sample(n=len(dataset_true), random_state=random_state)
            # Balanced classes are merged, randomized, and re-indexed
            dataset_balanced = pd.concat([dataset_true, dataset_false_sampled, dataset_mid]).sample(frac=1, random_state=random_state).reset_index(drop=drop)

        # Save balanced dataset to a new CSV
        filename = csv_path.split('.')[0] + "_balanced.csv"
        dataset_balanced.to_csv(filename, index=index)
        print('Balanced dataset saved to ' + os.path.abspath(filename))
        
        return dataset_balanced

def smote(x_train, y_train, dataset, random_state=42, index=False):
    """
    Balance a dataset using Synthetic Minority Oversampling Technique (SMOTE).

    Args:
        x_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Labels corresponding to x_train.
        dataset (str): Path to the original CSV file (used to derive output name).
        random_state (int, optional): Seed for reproducibility. Defaults to 42.
        index (bool, optional): Whether to include the DataFrame index when saving. Defaults to False.

    Returns:
        tuple(pd.DataFrame, pd.Series): The oversampled features and labels.

    Example:
        >>> x_res, y_res = smote(x_train, y_train, "data.csv")
    """
    sm = SMOTE(random_state=random_state)
    # Generate synthetic samples to balance minority class
    x, y = sm.fit_resample(x_train, y_train)

    # Combine resampled features and labels for output
    df_balanced = pd.concat([x, y], axis=1)

    # Construct a new filename and save balanced dataset
    filename = dataset.split('.')[0] + "_balanced.csv"
    df_balanced.to_csv(filename, index=index)
    print('Balanced dataset saved to ' + os.path.abspath(filename))

    return x, y

def clean_missing(csv_path, index_col=0, index=False):
    """
    Remove rows with missing values and report removal statistics.

    Args:
        csv_path (str): Path to the CSV file.
        index_col (int, optional): Column to use as DataFrame index. Defaults to 0.
        index (bool, optional): Whether to include the DataFrame index when saving. Defaults to False.

    Returns:
        tuple(pd.DataFrame, dict, str):
            - Cleaned DataFrame with no missing entries.
            - Dictionary with keys "missing data points" and "% missing data points".
            - Filename of the saved cleaned CSV.

    Example:
        >>> df_clean, stats, fname = clean_missing("data.csv")
        >>> print(stats)
        {'missing data points': 42, '% missing data points': 0.07}
    """
    dataset = pd.read_csv(csv_path, index_col=index_col)

    # Calculate number and fraction of rows with any missing values
    missing = len(dataset[dataset.isnull().any(axis=1)])
    missing_pct = missing / len(dataset)

    missing_stats = { "missing data points": missing,
                      "% missing data points": missing_pct,
                    }

    # Remove rows that contain any null values
    dataset_clean = dataset[dataset.notnull().all(axis=1)].copy()
    
    # Save cleaned dataset
    filename = csv_path.split('.')[0] + "_clean.csv"
    dataset_clean.to_csv(filename, index=index)
    print('Clean dataset saved to ' + os.path.abspath(filename))
          
    return dataset_clean, missing_stats, filename

def split_dataset(dataset, test_size=0.3, random_state=42):
    """
    Split features and target from a DataFrame and partition into train/test sets.

    Args:
        dataset (pd.DataFrame): DataFrame with features and target in the last column.
        test_size (float, optional): Proportion of data to reserve for testing. Defaults to 0.3.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
            - x_train: Training features.
            - x_test: Testing features.
            - y_train: Training labels.
            - y_test: Testing labels.
    """
    # Target is assumed to be the last column
    labels = dataset.columns[-1]
    x, y = dataset.drop(columns=[labels]), dataset[labels]

    # Stratify split to preserve class proportions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)

    return x_train, x_test, y_train, y_test

def balance_stats(y):
    """
    Calculate the class distribution for a binary label series.

    Args:
        y (pd.Series): Binary labels, where 1 is the positive class and 0 the negative.

    Returns:
        dict: Proportion of samples in each class, with keys "1" and "0".
    """
    p1 = (y==1).mean().item()
    p0 = (y==0).mean().item()
    return {"1": p1, "0": p0}

def binarize_labels(dataset, labels, true_labels, false_labels):
    """
    Binarize categorical labels in a DataFrame column.

    Args:
        dataset (pd.DataFrame): DataFrame containing the labels.
        labels (str): Name of the column to binarize.
        true_labels (Any): Value in `labels` to map to 1.
        false_labels (Any): Value in `labels` to map to 0.

    Returns:
        pd.Series: Series of binary labels where `true_labels` → 1 and `false_labels` → 0.
    """
    # Create mapping from original labels to binary values
    label_map = {
        str(true_labels): 1,
        str(false_labels): 0,
    }
    return dataset[labels].map(label_map)

def str_to_num(x, drop_first=True):
    """
    Convert categorical features into one-hot encoded numeric variables.

    Args:
        x (pd.DataFrame or pd.Series): Input data with categorical columns.
        drop_first (bool, optional): Drop the first category level to avoid multicollinearity. Defaults to True.

    Returns:
        pd.DataFrame: One-hot encoded DataFrame.
    """
    # One-hot encode categorical inputs
    x = pd.get_dummies(x, drop_first=drop_first)
    return x

def binarize_encode(csv_path, true_labels, false_labels, index_col=0, index=False):
    """
    Load a CSV, sanitize column names, binarize the target column, one-hot encode features,
    and save the resulting DataFrame.

    Args:
        csv_path (str): Path to the input CSV file.
        true_labels (Any): Value in the target column to map to 1.
        false_labels (Any): Value in the target column to map to 0.
        index_col (int, optional): Column to use as DataFrame index. Defaults to 0.
        index (bool, optional): Whether to include the DataFrame index when saving. Defaults to False.

    Returns:
        tuple(pd.DataFrame, str): 
            - Encoded DataFrame with binary labels and one-hot features.
            - Filename of the saved CSV.

    Example:
        >>> df_enc, fname = binarize_encode("data.csv", "yes", "no")
    """
    dataset = pd.read_csv(csv_path, index_col=index_col)

    # Sanitize column names by replacing non-word characters with underscores
    dataset.columns = [re.sub(r"[^\w]+", "_", c).strip("_") for c in dataset.columns]

    # Separate features and raw target
    x, y = dataset.iloc[:, :-1], dataset.columns[-1]

    # Binarize target and encode categorical features
    y_binarized = binarize_labels(dataset, y, true_labels, false_labels)
    x_encoded = str_to_num(x)

    # Combine encoded features and binary labels
    dataset_encoded = pd.concat([x_encoded, y_binarized], axis=1)

    # Save to new CSV
    filename = csv_path.split('.')[0] + "_encoded.csv"
    dataset_encoded.to_csv(filename, index=index)
    print('Encoded dataset saved to ' + os.path.abspath(filename))

    return dataset_encoded, filename