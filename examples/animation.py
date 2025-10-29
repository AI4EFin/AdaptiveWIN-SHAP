import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Example data ---
np.random.seed(0)
x = np.linspace(0, 20, 200)
y = np.sin(x) + 0.2 * np.random.randn(len(x))

# --- Params ---
window_size = 20
step = 1
interval_ms = 50

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
ax.set_title("Sliding Window Animation", fontsize=14)

ax.plot(x, y, color='lightgray', lw=1)
(line_window,) = ax.plot([], [], lw=2)         # window trace
(marker_start,) = ax.plot([], [], 'o', ms=6)    # start point
(marker_end,) = ax.plot([], [], 'o', ms=6)      # end point

def init():
    line_window.set_data([], [])
    marker_start.set_data([], [])
    marker_end.set_data([], [])
    return line_window, marker_start, marker_end

def update(frame):
    start = frame * step
    end = min(start + window_size, len(x))
    # window segment
    line_window.set_data(x[start:end], y[start:end])
    # SINGLE POINTS: pass sequences, not scalars
    marker_start.set_data([x[start]], [y[start]])
    marker_end.set_data([x[end - 1]], [y[end - 1]])
    return line_window, marker_start, marker_end

# frames: include last valid window position
frames = max(1, (len(x) - window_size) // step + 1)

ani = FuncAnimation(
    fig, update, init_func=init, frames=frames, blit=True, interval=interval_ms, repeat=True
)

plt.show()
