# Data Engineering Spectrogram GUI

This repository contains a simple Tkinter application for loading, slicing and saving spectrogram data stored as `.npy` files.

Run the GUI with:

```bash
python main.py
```

Load one or more numpy arrays and they will be pooled immediately using the configured window size. Crop each spectrogram using the mouse and `c` key, then save all windows as a new array.

### Accepted array layouts

The loader can handle spectrograms stored individually as 2‑D arrays as well as
files that stack multiple spectrograms along the **first** axis. A stacked file
must therefore have the shape ``(n_spects, freq, time)``. The application splits
such arrays automatically and loads each spectrogram individually.
