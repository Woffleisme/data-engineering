import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

from spect_load import load_spects
from spect_slice import average_pooling, rescale, create_windows


class SpectAnalyzer:
    """GUI application for interactive spectrogram slicing."""

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Spect Analyzer")

        load_btn = tk.Button(master, text="Load NPY Files", command=self.load_files)
        load_btn.pack(pady=5)

        opts = tk.Frame(master)
        opts.pack()

        tk.Label(opts, text="Pooling window rows:").grid(row=0, column=0)
        self.pool_rows = tk.Entry(opts, width=5)
        self.pool_rows.insert(0, "8")
        self.pool_rows.grid(row=0, column=1)

        tk.Label(opts, text="cols:").grid(row=0, column=2)
        self.pool_cols = tk.Entry(opts, width=5)
        self.pool_cols.insert(0, "2")
        self.pool_cols.grid(row=0, column=3)

        pool_btn = tk.Button(opts, text="Apply Pooling", command=self.apply_pooling)
        pool_btn.grid(row=0, column=4, padx=5)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("key_press_event", self.on_key)

        save_btn = tk.Button(master, text="Create Windows & Save", command=self.save_windows)
        save_btn.pack(pady=5)

        self.spects: list[np.ndarray] = []
        self.processed: list[np.ndarray] = []
        self.index = 0
        self.selector: RectangleSelector | None = None

    # --- Data loading and pooling -------------------------------------------------
    def load_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("NumPy arrays", "*.npy")])
        if not paths:
            return
        self.spects = load_spects(list(paths))
        self.processed = self.spects.copy()
        self.index = 0
        # automatically pool for easier viewing
        self.apply_pooling()

    def apply_pooling(self) -> None:
        if not self.spects:
            messagebox.showinfo("Info", "Load files first")
            return
        try:
            win = (int(self.pool_rows.get()), int(self.pool_cols.get()))
        except ValueError:
            messagebox.showerror("Error", "Invalid window size")
            return
        pooled = []
        for spect in self.spects:
            if spect.shape[0] < win[0] or spect.shape[1] < win[1]:
                messagebox.showerror(
                    "Error", f"Spect shape {spect.shape} too small for window {win}"
                )
                return
            pooled.append(average_pooling(spect, win))
        self.processed = pooled
        self.index = 0
        self.display_current()

    # --- Display and interaction --------------------------------------------------
    def display_current(self) -> None:
        self.ax.clear()
        if not self.processed:
            self.canvas.draw()
            return
        spect = self.processed[self.index]
        self.ax.imshow(rescale(spect).T, origin="lower")
        self.ax.set_title(f"Spect {self.index + 1}/{len(self.processed)}")

        if self.selector:
            self.selector.disconnect_events()
        self.selector = RectangleSelector(
            self.ax,
            onselect=lambda *args: None,
            drawtype="box",
            useblit=True,
            button=[1],
            interactive=True,
        )
        self.canvas.draw()

    def crop_current(self) -> None:
        if not self.selector:
            return
        x1, x2, y1, y2 = map(int, self.selector.extents)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        spect = self.processed[self.index]
        self.processed[self.index] = spect[y1:y2, x1:x2]
        self.display_current()

    def on_key(self, event) -> None:
        if event.key == "right":
            self.index = (self.index + 1) % len(self.processed)
            self.display_current()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.processed)
            self.display_current()
        elif event.key == "c":
            self.crop_current()

    # --- Window creation and saving ----------------------------------------------
    def save_windows(self) -> None:
        if not self.processed:
            return
        window_shape = (24, 96)
        window_step = (4, 8)
        parts: list[np.ndarray] = []
        for spect in self.processed:
            if spect.shape[0] < window_shape[0] or spect.shape[1] < window_shape[1]:
                continue
            parts.extend(create_windows(spect, window_shape, window_step))
        if not parts:
            messagebox.showerror("Error", "No windows created")
            return
        result = np.stack(parts)
        path = filedialog.asksaveasfilename(
            defaultextension=".npy", filetypes=[("NumPy arrays", "*.npy")]
        )
        if path:
            np.save(path, result)
            messagebox.showinfo("Saved", f"Saved {result.shape[0]} windows to {path}")


def main() -> None:
    root = tk.Tk()
    app = SpectAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
