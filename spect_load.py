import numpy as np
from typing import List

def load_spects(paths: List[str]) -> List[np.ndarray]:
    """Load multiple ``.npy`` files and return list of 2D spectrogram arrays.

    The application expects each spectrogram to be a 2D array.  If the loaded
    data is 3‑D it may contain multiple spectrograms stacked either on the first
    axis ``(n_spects, time, freq)`` or the last axis ``(freq, time, n_spects)``.
    In those cases the array is split along the detected stack axis and each
    slice is returned individually.
    """

    spects: List[np.ndarray] = []
    for path in paths:
        data = np.load(path)

        if data.ndim == 3:
            # Determine if spectrograms are stacked on the first or last axis.
            first_is_stack = data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]
            if first_is_stack:
                for i in range(data.shape[0]):
                    spects.append(data[i, ...])
            else:
                for i in range(data.shape[-1]):
                    spects.append(data[..., i])
        else:
            spects.append(data)

    return spects
