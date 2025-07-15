import numpy as np
from typing import List

def load_spects(paths: List[str]) -> List[np.ndarray]:
    """Load multiple ``.npy`` files and return list of 2D spectrogram arrays.

    The application expects one spectrogram at a time.  If the loaded
    data is 3‑D it may contain multiple spectrograms stacked  on the first
    axis, the shape of the spects is n_spectrograms x freq x time. so for example 
    37 x 1024 x 480 would mean 37 spectrograms with 1024 frequency bins and 480 time steps. The input shape will
    always have 1024 frequency bins and 480 time steps, so the first dimension
    will always be the number of spectrograms.

    Args:
        paths: List of file paths to load.

    Returns:
        List of 2D spectrogram arrays (freq x time).
    """
    spects = []
    
    for path in paths:
        data = np.load(path)
        
        # If 3D, split into individual 2D spectrograms
        if data.ndim == 3:
            for i in range(data.shape[0]):
                spects.append(data[i])
        # If 2D, add as single spectrogram
        elif data.ndim == 2:
            spects.append(data)
    
    return spects

