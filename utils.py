import numpy as np
from typing import Optional, Union, Tuple


def euclidean_distance(x: np.array, y: np.array):
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """


    return np.sqrt(np.sum(np.power(x - y, 2)))


def euclidean_similarity(x: np.array, y: np.array):
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    return 1 / (1 + np.sqrt(np.sum(np.power(x - y, 2))))


def pearson_similarity(x: np.array, y: np.array):
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    a = np.sum((x - np.mean(x))*(y - np.mean(y)))
    b = np.sqrt(np.sum(np.power(x - np.mean(x), 2))*np.sum(np.power(y - np.mean(y), 2)))
    return a / b


def apk(actual: np.array, predicted: np.array, k: int = 10):
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)


    return score / min(len(actual), k)


def mapk(actual: np.array, predicted: np.array, k: int = 10):
    """
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
