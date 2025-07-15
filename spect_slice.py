import numpy as np
from skimage import exposure, util
from typing import Iterable, List, Tuple


def average_pooling(data2d: np.ndarray, window_size: Tuple[int, int]) -> np.ndarray:
    """Average pool ``data2d`` using ``window_size``."""
    windows = util.view_as_blocks(data2d, window_size)
    avg = np.average(windows, axis=(2, 3))
    return avg / np.max(avg)


def rescale(spect: np.ndarray, a: int = 2, b: int = 98) -> np.ndarray:
    """Rescale ``spect`` for display using percentile stretch."""
    p2, p98 = np.percentile(spect, (a, b))
    return exposure.rescale_intensity(spect, in_range=(p2, p98))


def create_windows(spect: np.ndarray, window_shape: Tuple[int, int], step: Tuple[int, int]) -> List[np.ndarray]:
    """Slice spect into overlapping windows."""
    windows = util.view_as_windows(spect, window_shape=window_shape, step=step)
    return list(windows.reshape((-1, *window_shape)))
