import numpy as np
from typing import List

def load_spects(paths: List[str]) -> List[np.ndarray]:
    """Load multiple ``.npy`` files and return list of arrays."""
    return [np.load(p) for p in paths]
