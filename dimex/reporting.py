"""
Reporting utilities for model evaluation.

This module provides:
- cm: Plot and display a confusion matrix for given true and predicted labels.
"""

__author__ = "Lucas Campagnaro"

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def cm(y, y_pred, cmap="Blues", labels=[0, 1], values_format='d'):
    """
    Plot a confusion matrix for classification results.

    Args:
        y (array-like): True labels.
        y_pred (array-like): Predicted labels.
        cmap (str, optional): Matplotlib colormap. Defaults to "Blues".
        labels (list, optional): Label values to index the matrix. Defaults to [0, 1].
        values_format (str, optional): Format code for cell values. Defaults to 'd'.

    Returns:
        None: Displays the confusion matrix plot.
    """
    matrix = confusion_matrix(y, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(cmap=cmap, values_format='d')
    plt.show()
    