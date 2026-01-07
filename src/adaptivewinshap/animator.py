import numpy as np
import matplotlib.pyplot as plt


class SlidingWindowAnimator:
    """
    Single-plot version of the sliding window animator.
    Shows:
      - Time series
      - Current window I_{k+1}
      - Candidate splits J_k
      - A_k (red dashed) and B_k (blue dashed)
      - Debug text (io, k, l, I_k±1, J_k, SupLR, crit) shown ABOVE the plot

    Can save as GIF/MP4 (HiDPI safe) and replay.
    """

    def __init__(self, x, y, title="QLR Sliding Window Debugger", pause=0.05, record=True):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.pause = float(pause)
        self.record = bool(record)

        # Recording (for GIF/MP4 or replay)
        self.states = []
        self.frames = []

        plt.ion()
        self.fig, self.ax_ts = plt.subplots(figsize=(9, 4))

        self.fig.patch.set_alpha(0.0)
        self.ax_ts.set_facecolor("none")

        # --- Left/main: time series
        # self.ax_ts.set_title(title)
        self.ax_ts.set_xlim(self.x.min(), self.x.max())
        y_pad = 0.05 * (self.y.max() - self.y.min() + 1e-9)
        self.ax_ts.set_ylim(self.y.min() - y_pad, self.y.max() + y_pad)
        self.ax_ts.spines["top"].set_visible(False)
        self.ax_ts.spines["right"].set_visible(False)

        (self.line_all,) = self.ax_ts.plot(self.x, self.y, lw=1, color="lightgray", zorder=1)
        (self.line_win,) = self.ax_ts.plot([], [], lw=2, color="tab:blue", zorder=3)
        (self.pt_start,) = self.ax_ts.plot([], [], 'o', ms=6, color="tab:red", zorder=4)
        (self.pt_end,) = self.ax_ts.plot([], [], 'o', ms=6, color="tab:orange", zorder=4)

        self.split_ticks = None
        self._akbk_handles = []

        # --- Debug text (outside, above the plot)
        self.debug_text = self.fig.text(
            0.5, -0.01,  # x=center, y=bottom (in figure coords)
            "",  # initial text
            ha="center", va="bottom",  # anchored at bottom
            fontsize=9.5,
            color="black",  # adjust to white if overlaying on dark bg
            bbox=dict(boxstyle="square,pad=0", fc="none", ec="none", alpha=0.0)
        )

        self.fig.tight_layout()

    # ---------- helpers ----------
    def _x_val(self, idx):
        idx = int(np.clip(idx, 0, len(self.x) - 1))
        return self.x[idx]

    def _clear_split_ticks(self):
        if self.split_ticks is not None:
            for l in self.split_ticks:
                try:
                    l.remove()
                except Exception:
                    pass
        self.split_ticks = None

    def _draw_candidate_splits(self, J_abs):
        self._clear_split_ticks()
        if J_abs is None or len(J_abs) == 0:
            return
        y0, y1 = self.ax_ts.get_ylim()
        h = 0.05 * (y1 - y0)
        ticks = []
        for j in np.asarray(J_abs, dtype=float):
            t = self.ax_ts.plot(
                [self._x_val(j), self._x_val(j)], [y0, y0 + h], lw=1, alpha=0.75, color="gray"
            )[0]
            ticks.append(t)
        self.split_ticks = ticks

    def _clear_akbk(self):
        for h in self._akbk_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._akbk_handles = []

    def _draw_akbk(self, start_idx, end_idx, tau):
        """Draw A_k (red dashed) and B_k (blue dashed) along baseline."""
        self._clear_akbk()
        if tau is None:
            return

        start = int(max(0, start_idx))
        end = int(min(len(self.x), end_idx))
        tau = int(np.clip(tau, start, end))

        y0, y1 = self.ax_ts.get_ylim()
        yA = y0 + 0.02 * (y1 - y0)

        if tau > start:
            hA, = self.ax_ts.plot(
                [self._x_val(start), self._x_val(tau)],
                [yA, yA],
                linestyle="-",
                linewidth=3,
                alpha=0.95,
                color="crimson",
            )
            self._akbk_handles.append(hA)
        if end > tau:
            hB, = self.ax_ts.plot(
                [self._x_val(tau), self._x_val(end - 1)],
                [yA, yA],
                linestyle="-",
                linewidth=3,
                alpha=0.95,
                color="royalblue",
            )
            self._akbk_handles.append(hB)

    def _rasterize(self, with_alpha=False):
        """HiDPI-safe capture of current canvas as np.array.

        Args:
            with_alpha: If True, return RGBA (H,W,4). If False, return RGB (H,W,3).
        """
        self.fig.canvas.draw()
        try:
            rgba = np.asarray(self.fig.canvas.buffer_rgba())  # (H,W,4)
            if with_alpha:
                return rgba.copy()
            return rgba[..., :3].copy()
        except Exception:
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height()
            return buf.reshape(h, w, 3)

    # ---------- public API ----------
    def update(
        self,
        io,
        start_idx,
        end_idx,
        J_abs=None,
        sup_lr=None,
        crit=None,
        k=None,
        l=None,
        tau=None,
        n_k_plus1=None,
        n_k=None,
        n_k_minus1=None,
        J_start=None,
        J_end=None,
    ):
        """Update plot for current window and debug info."""
        # --- Left: I_{k+1}
        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(self.x), end_idx))

        if end_idx <= start_idx:
            self.line_win.set_data([], [])
            self.pt_start.set_data([], [])
            self.pt_end.set_data([], [])
        else:
            xs = self.x[start_idx:end_idx]
            ys = self.y[start_idx:end_idx]
            self.line_win.set_data(xs, ys)
            self.pt_start.set_data([xs[0]], [ys[0]])
            self.pt_end.set_data([xs[-1]], [ys[-1]])

        self._draw_candidate_splits(J_abs)
        self._draw_akbk(start_idx, end_idx, tau)

        # --- Debug text (top of figure)
        lines = [f"$t$={io} | $k$={k} |"]
        if (n_k_plus1 is not None) and (n_k is not None) and (n_k_minus1 is not None):
            Ik1_s = max(0, io - int(n_k_plus1))
            Ik1_e = io
            Ik_s = max(0, io - int(n_k))
            Ik_e = io
            Ikm1_s = max(0, io - int(n_k_minus1))
            Ikm1_e = io
            lines.append(
                f"$I_{{k+1}}$=[{Ik1_s}, {Ik1_e}] | $I_k$=[{Ik_s}, {Ik_e}] | $I_{{k-1}}$=[{Ikm1_s}, {Ikm1_e}] | "
            )
        if (J_start is not None) and (J_end is not None):
            lines.append(f"$J_k$=[{int(J_start)}, {int(J_end)}] | ")
        if sup_lr is not None:
            lines.append(f"$T_{{I_k}}$={sup_lr:.3f} | ")
        if crit is not None:
            lines.append(f"$\mathfrak{{z}}_{{I_k}}^{{\circ }}(\\alpha)$={crit:.3f}")
        self.debug_text.set_text("  ".join(lines))

        # --- Draw & optionally record
        self.fig.canvas.draw_idle()
        plt.pause(self.pause)

        if self.record:
            state = dict(
                io=io,
                start_idx=start_idx,
                end_idx=end_idx,
                J_abs=np.array(J_abs) if J_abs is not None else np.array([], dtype=int),
                sup_lr=sup_lr,
                crit=crit,
                k=k,
                l=l,
                tau=tau,
                n_k_plus1=n_k_plus1,
                n_k=n_k,
                n_k_minus1=n_k_minus1,
                J_start=J_start,
                J_end=J_end,
            )
            self.states.append(state)
            self.frames.append(self._rasterize())

    # ---------- re-render for replay/save ----------
    def _render_state(self, st):
        self.update(
            io=st["io"],
            start_idx=st["start_idx"],
            end_idx=st["end_idx"],
            J_abs=st["J_abs"],
            sup_lr=st["sup_lr"],
            crit=st["crit"],
            k=st["k"],
            l=st["l"],
            tau=st.get("tau", None),
            n_k_plus1=st.get("n_k_plus1", None),
            n_k=st.get("n_k", None),
            n_k_minus1=st.get("n_k_minus1", None),
            J_start=st.get("J_start", None),
            J_end=st.get("J_end", None),
        )

    # ---------- export ----------
    def save(self, path="frames/frame.png", fps=16, boomerang=True, codec="h264", transparent=True):
        """Save as .gif, .mp4, .webm, or .png sequence.

        Args:
            path: Output file path. Use .webm for transparent video, or a path
                  ending in .png to save as numbered PNG sequence (e.g., "frames/frame.png"
                  will create frames/frame_0001.png, frames/frame_0002.png, etc.)
            fps: Frames per second.
            boomerang: If True, play forward then backward.
            codec: Video codec. Use "h264" for mp4, "vp9" for webm with transparency.
            transparent: If True, preserve alpha channel (requires .webm or .png output).
        """
        if not (self.frames or self.states):
            raise RuntimeError("Nothing to save.")

        ext = path.lower().rsplit(".", 1)[-1]

        # Re-render frames with alpha if needed and we have states
        if transparent and self.states:
            frames = []
            old_record = self.record
            self.record = False  # Don't re-record while re-rendering
            for st in self.states:
                self._render_state(st)
                frames.append(self._rasterize(with_alpha=True))
            self.record = old_record
        else:
            frames = self.frames if self.frames else None

        if frames:
            seq = frames
            if boomerang and len(seq) > 1:
                seq = seq + seq[-2:0:-1]

            # PNG sequence for full transparency support
            if ext == "png":
                import imageio.v3 as iio
                import os
                base, _ = os.path.splitext(path)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                for i, frame in enumerate(seq):
                    frame_path = f"{base}_{i:04d}.png"
                    iio.imwrite(frame_path, frame)
                return

            if ext == "gif":
                import imageio.v3 as iio
                iio.imwrite(path, seq, duration=1.0 / fps, loop=0)
                return

            if ext == "webm" and transparent:
                import imageio.v3 as iio
                # VP9 codec supports alpha channel in WebM
                iio.imwrite(path, seq, fps=fps, codec="vp9",
                            output_params=["-pix_fmt", "yuva420p"])
                return

            if ext in ("mp4", "mov", "mkv", "webm"):
                import imageio.v3 as iio
                iio.imwrite(path, seq, fps=fps, codec=codec)
                return

        # fallback writer
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
        with writer.saving(self.fig, path, dpi=self.fig.dpi):
            states = self.states
            if boomerang and len(states) > 1:
                states = states + states[-2:0:-1]
            for st in states:
                self._render_state(st)
                writer.grab_frame()

    def close(self):
        plt.ioff()
        plt.show(block=False)
