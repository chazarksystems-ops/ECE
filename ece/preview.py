"""Interactive CPU preview for 2D Mohr simulations."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import numpy as np

from .config import load_config
from .matrix import parse_matrix_values
from .step_mohr import MohrState, seed_state, step


_SPECIES_COLORS = ("#f0a35e", "#78c4a3", "#e36f6f", "#78a9d1", "#d9c86c", "#b48bd0")


class MohrPreview:
    def __init__(self, root: tk.Tk, cfg, backend: str = "cpu", max_frames: int | None = None):
        self.root = root
        self.cfg = cfg
        self.backend = backend
        self.max_frames = max_frames if max_frames is not None else int(cfg.io.get("frames", 1000))
        self.state = seed_state(cfg)
        self.running = False
        self.speed = tk.IntVar(value=1)
        self.status = tk.StringVar()
        self.particle_items: list[int] = []

        root.title("ECE | Mohr Preview")
        root.minsize(780, 620)
        layout = ttk.Frame(root, padding=12)
        layout.pack(fill="both", expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            layout,
            width=680,
            height=680,
            background="#10181b",
            highlightthickness=1,
            highlightbackground="#34464b",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        controls = ttk.Frame(layout, padding=(14, 0, 0, 0))
        controls.grid(row=0, column=1, sticky="ns")
        ttk.Label(controls, text="MOHR / 2D", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(2, 16))
        ttk.Label(controls, textvariable=self.status, width=24).pack(anchor="w", pady=(0, 12))
        self.play_button = ttk.Button(controls, text="Start", command=self.toggle_running)
        self.play_button.pack(fill="x", pady=3)
        ttk.Button(controls, text="Step", command=self.step_once).pack(fill="x", pady=3)
        ttk.Button(controls, text="Reset", command=self.reset).pack(fill="x", pady=3)
        ttk.Button(controls, text="Edit matrix", command=self.edit_matrix).pack(fill="x", pady=(3, 14))
        ttk.Label(controls, text="Steps per update (1-4)").pack(anchor="w")
        ttk.Scale(controls, from_=1, to=4, variable=self.speed, orient="horizontal").pack(fill="x", pady=(3, 12))
        ttk.Label(controls, text="Species", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 6))
        for species in range(cfg.species_count):
            color = _SPECIES_COLORS[species % len(_SPECIES_COLORS)]
            row = ttk.Frame(controls)
            row.pack(anchor="w", fill="x", pady=1)
            swatch = tk.Canvas(row, width=10, height=10, background=color, highlightthickness=0)
            swatch.pack(side="left", padx=(0, 7))
            ttk.Label(row, text=str(species)).pack(side="left")

        self.canvas.bind("<Configure>", lambda _event: self.draw())
        root.bind("<space>", lambda _event: self.toggle_running())
        root.bind("n", lambda _event: self.step_once())
        root.bind("r", lambda _event: self.reset())
        root.after(0, self.draw)
        root.after(16, self.tick)

    def toggle_running(self) -> None:
        if self.state.frame >= self.max_frames:
            self.reset()
        self.running = not self.running
        self.play_button.configure(text="Pause" if self.running else "Start")

    def step_once(self) -> None:
        self.state = self._step()
        self.draw()

    def _step(self) -> MohrState:
        if self.backend == "cuda":
            from .cuda_mohr import step_cuda

            return step_cuda(self.state, self.cfg)
        if self.backend == "wgpu":
            from .wgpu_mohr import step_wgpu

            return step_wgpu(self.state, self.cfg)
        return step(self.state, self.cfg)

    def reset(self) -> None:
        self.running = False
        self.play_button.configure(text="Start")
        self.state = seed_state(self.cfg)
        self.draw()

    def tick(self) -> None:
        if self.running:
            remaining = self.max_frames - self.state.frame
            for _ in range(min(max(1, self.speed.get()), remaining)):
                self.state = self._step()
            if self.state.frame >= self.max_frames:
                self.running = False
                self.play_button.configure(text="Restart")
            self.draw()
        self.root.after(16, self.tick)

    def draw(self) -> None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        positions = self.state.pos / self.cfg.world
        if len(self.particle_items) != len(positions):
            self.canvas.delete("particle")
            self.particle_items.clear()
            for species in self.state.types:
                color = _SPECIES_COLORS[int(species) % len(_SPECIES_COLORS)]
                self.particle_items.append(
                    self.canvas.create_oval(0, 0, 0, 0, fill=color, outline="", tags="particle")
                )
        for item, position in zip(self.particle_items, positions):
            x = float(position[0]) * width
            y = (1.0 - float(position[1])) * height
            radius = 3.0
            self.canvas.coords(item, x - radius, y - radius, x + radius, y + radius)
        self.status.set(
            f"{self.backend.upper()} | frame {self.state.frame}/{self.max_frames}\n"
            f"{len(self.state.pos)} particles"
        )

    def edit_matrix(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Species interaction matrix")
        window.transient(self.root)
        window.grab_set()
        size = self.cfg.species_count
        ttk.Label(window, text="Force from row species to column species").grid(
            row=0, column=0, columnspan=size + 1, padx=12, pady=(12, 8), sticky="w"
        )
        for species in range(size):
            ttk.Label(window, text=str(species)).grid(row=1, column=species + 1, padx=3, pady=3)
            ttk.Label(window, text=str(species)).grid(row=species + 2, column=0, padx=5, pady=3)
        variables: list[list[tk.StringVar]] = []
        for row in range(size):
            variable_row = []
            for column in range(size):
                value = tk.StringVar(value=f"{self.cfg.matrix[row, column]:.3f}")
                ttk.Entry(window, textvariable=value, width=7, justify="center").grid(
                    row=row + 2, column=column + 1, padx=2, pady=2
                )
                variable_row.append(value)
            variables.append(variable_row)

        def apply_matrix() -> None:
            try:
                matrix = parse_matrix_values(
                    [[value.get() for value in row] for row in variables], size
                )
            except ValueError as exc:
                messagebox.showerror("Invalid matrix", str(exc), parent=window)
                return
            self.cfg.matrix[:] = matrix
            window.destroy()

        ttk.Button(window, text="Apply", command=apply_matrix).grid(
            row=size + 2, column=0, columnspan=size + 1, padx=12, pady=12, sticky="e"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive CPU preview for Mohr Particle Life")
    parser.add_argument("config", type=Path, help="path to a TOML simulation config")
    parser.add_argument("--particles", type=int, help="override configured particle count")
    parser.add_argument("--frames", type=int, help="override configured frame limit")
    parser.add_argument("--backend", choices=("cpu", "cuda", "wgpu"), default="cpu")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    max_frames = args.frames if args.frames is not None else int(cfg.io.get("frames", 1000))
    if max_frames < 1:
        parser.error("frame limit must be at least 1")
    if cfg.rules != ["mohr"]:
        parser.error('currently supports configs with rules = ["mohr"]')
    missing = {"r_max", "beta"} - cfg.mohr.keys()
    if args.particles is None:
        missing |= {"particles"} - cfg.mohr.keys()
    if missing:
        parser.error(f"Mohr config is missing: {', '.join(sorted(missing))}")
    if args.particles is not None:
        if args.particles < 1:
            parser.error("--particles must be at least 1")
        cfg.mohr["particles"] = args.particles
    if args.backend == "cuda":
        from .cuda_mohr import cuda_available

        if not cuda_available():
            parser.error("CUDA is unavailable; install the cuda extra and check the driver")
    elif args.backend == "wgpu":
        from .wgpu_mohr import webgpu_available

        if not webgpu_available():
            parser.error("WebGPU is unavailable; install the webgpu extra and check the adapter")

    root = tk.Tk()
    MohrPreview(root, cfg, backend=args.backend, max_frames=max_frames)
    root.mainloop()


if __name__ == "__main__":
    main()