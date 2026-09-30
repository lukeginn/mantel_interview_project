import numpy as np
from sklearn.datasets import load_wine


def make_random_colours(n_samples=10, seed=42):
    """Random RGB colours, one per row, with values between 0 and 1."""
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, 3))


def load_wine_dataset():
    """The UCI wine dataset: 178 wines, 13 chemical measurements, 3 grape cultivars.

    Returns the measurements in their original units, the cultivar of each wine,
    and the column names.
    """
    wine = load_wine()
    return wine.data, wine.target, wine.feature_names


def standardise(data):
    """Rescale each column to mean 0 and standard deviation 1.

    The SOM compares rows by distance, so columns on very different scales
    (e.g. proline in the hundreds, hue around 1) must be put on the same scale
    first, or the largest one dominates.
    """
    return (data - data.mean(axis=0)) / data.std(axis=0)
