"""
gui_analyze.py — tkinter GUI front-end for analyze_chamber_ramps.run_analysis.

Usage:
    python gui_analyze.py

Select a CSV file or a folder that contains the chamber log CSV, then click
"Run Analysis". Progress and output are shown in the log pane. Outputs
(CSVs + figures/) are written next to the input CSV unless an explicit output
directory is chosen. All analysis parameters use their built-in defaults.
"""
from __future__ import annotations

import queue
import re
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from analyze_chamber_ramps import (
    resolve_csv_input,
    run_analysis,
)

_NUM_GROUPS = 6
_NUM_DUTS = 15
_EXCL_TOKEN_RE = re.compile(r"^G(\d+)D(\d+)$", re.IGNORECASE)
_INTERMEDIATE_CSVS = (
    "ramp_metrics.csv",
    "soak_segment_summary.csv",
    "soak_dwell_times.csv",
    "soak_boxplot_resistance_long.csv",
)


class _QueueStream:
    """Thread-safe stdout/stderr redirector that feeds text into a queue."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        self._q = q

    def write(self, text: str) -> None:
        if text:
            self._q.put(text)

    def flush(self) -> None:
        pass


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Chamber Ramp Analysis")
        self.minsize(700, 640)
        self.resizable(True, True)

        self._q: queue.Queue[str] = queue.Queue()
        self._running = False
        self._status_label: tk.Label  # assigned in _build_ui

        self._archive_var = tk.BooleanVar(value=True)

        # Exclusion state – one BooleanVar per group, 6×15 for DUTs
        self._grp_vars: list[tk.BooleanVar] = [tk.BooleanVar() for _ in range(_NUM_GROUPS)]
        self._dut_vars: list[list[tk.BooleanVar]] = [
            [tk.BooleanVar() for _ in range(_NUM_DUTS)] for _ in range(_NUM_GROUPS)
        ]
        self._sub_frames: list[ttk.Frame] = []  # populated in _build_exclusion_ui

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ── input / output paths ────────────────────────────────────────────
        path_frame = ttk.LabelFrame(self, text="Input / Output", padding=8)
        path_frame.pack(fill="x", padx=12, pady=(10, 4))
        path_frame.columnconfigure(1, weight=1)

        # Row 0 – input
        ttk.Label(path_frame, text="Input (CSV or Folder)").grid(
            row=0, column=0, sticky="w", **pad
        )
        self._input_var = tk.StringVar()
        self._input_var.trace_add("write", self._on_input_changed)
        ttk.Entry(path_frame, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", **pad
        )
        ttk.Button(path_frame, text="Browse…", width=8, command=self._browse_folder).grid(
            row=0, column=2, sticky="e", padx=8, pady=4
        )

        # ── exclusion panel ─────────────────────────────────────────────────
        self._build_exclusion_ui()

        # ── run controls ────────────────────────────────────────────────────
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=12, pady=(4, 2))

        self._run_btn = ttk.Button(ctrl_frame, text="Run Analysis", command=self._on_run)
        self._run_btn.pack(side="left", padx=(0, 8))
        ttk.Button(ctrl_frame, text="Clear Log", command=self._clear_log).pack(side="left")
        ttk.Checkbutton(
            ctrl_frame,
            text="Archive intermediate CSVs",
            variable=self._archive_var,
        ).pack(side="left", padx=(12, 0))

        self._status_label = tk.Label(
            ctrl_frame, text="Idle", fg="gray", anchor="e"
        )
        self._status_label.pack(side="right", padx=8)

        # ── log pane ────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self._log.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self._log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._log.configure(yscrollcommand=sb.set)

    def _build_exclusion_ui(self) -> None:
        excl_frame = ttk.LabelFrame(self, text="Exclusions", padding=8)
        excl_frame.pack(fill="x", padx=12, pady=(0, 4))

        # Top row: group checkboxes G1–G6
        grp_row = ttk.Frame(excl_frame)
        grp_row.pack(fill="x")

        for i in range(_NUM_GROUPS):
            cb = ttk.Checkbutton(
                grp_row,
                text=f"G{i + 1}",
                variable=self._grp_vars[i],
                command=lambda idx=i: self._toggle_group(idx),
            )
            cb.pack(side="left", padx=(0, 12))

        # Sub-frames for each group's DUT checkboxes (initially hidden)
        for i in range(_NUM_GROUPS):
            sub = ttk.Frame(excl_frame)
            # Don't pack yet — revealed only when group is checked
            for d in range(_NUM_DUTS):
                col = d % 5
                row = d // 5
                ttk.Checkbutton(
                    sub,
                    text=f"G{i + 1}D{d + 1}",
                    variable=self._dut_vars[i][d],
                ).grid(row=row, column=col, sticky="w", padx=4, pady=1)
            self._sub_frames.append(sub)

        # Save button
        save_row = ttk.Frame(excl_frame)
        save_row.pack(fill="x", pady=(6, 0))
        ttk.Button(save_row, text="Save Exclusions", command=self._save_exclusions).pack(side="left")

    # ------------------------------------------------------------------ browse

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self._input_var.set(path)


    # ------------------------------------------------------------------ exclusion helpers

    def _toggle_group(self, i: int) -> None:
        if self._grp_vars[i].get():
            for v in self._dut_vars[i]:
                v.set(True)
            self._sub_frames[i].pack(fill="x", padx=(24, 0), pady=(2, 0))
        else:
            self._sub_frames[i].pack_forget()

    def _save_exclusions(self) -> None:
        input_str = self._input_var.get().strip()
        if not input_str:
            messagebox.showerror("Input required", "Please select an input CSV file or folder first.")
            return

        input_path = Path(input_str).resolve()
        folder = input_path if input_path.is_dir() else input_path.parent
        excl_path = folder / "exclusion.csv"

        tokens: list[str] = []
        for g in range(_NUM_GROUPS):
            for d in range(_NUM_DUTS):
                if self._dut_vars[g][d].get():
                    tokens.append(f"G{g + 1}D{d + 1}")

        excl_path.write_text(", ".join(tokens), encoding="utf-8")

        msg = (
            f"Saved {len(tokens)} exclusion(s) to {excl_path.name}"
            if tokens
            else f"Cleared exclusions — {excl_path.name} is now empty."
        )
        self._append_log(f"[Exclusions] {msg}\n")
        messagebox.showinfo("Exclusions saved", msg)

    def _on_input_changed(self, *_: object) -> None:
        """Auto-load exclusion.csv when the input path changes."""
        input_str = self._input_var.get().strip()
        if not input_str:
            return
        input_path = Path(input_str).resolve()
        folder = input_path if input_path.is_dir() else input_path.parent
        excl_path = folder / "exclusion.csv"
        if excl_path.is_file():
            self._load_exclusions(excl_path)

    def _load_exclusions(self, excl_path: Path) -> None:
        # Reset everything first
        for i in range(_NUM_GROUPS):
            self._grp_vars[i].set(False)
            for v in self._dut_vars[i]:
                v.set(False)
            self._sub_frames[i].pack_forget()

        raw = excl_path.read_text(encoding="utf-8", errors="replace")
        for line in raw.splitlines():
            segment = line.split("#", 1)[0].strip()
            if not segment:
                continue
            for part in segment.replace(",", " ").split():
                m = _EXCL_TOKEN_RE.match(part.strip())
                if not m:
                    continue
                g, d = int(m.group(1)), int(m.group(2))
                if 1 <= g <= _NUM_GROUPS and 1 <= d <= _NUM_DUTS:
                    gi, di = g - 1, d - 1
                    self._dut_vars[gi][di].set(True)
                    if not self._grp_vars[gi].get():
                        self._grp_vars[gi].set(True)
                        self._sub_frames[gi].pack(fill="x", padx=(24, 0), pady=(2, 0))

    # ------------------------------------------------------------------ run

    def _on_run(self) -> None:
        if self._running:
            return

        input_str = self._input_var.get().strip()
        if not input_str:
            messagebox.showerror("Input required", "Please select an input CSV file or folder.")
            return

        self._running = True
        self._run_btn.configure(state="disabled", text="Running…")
        self._set_status("Running…", "blue")

        threading.Thread(
            target=self._analysis_thread,
            args=(input_str,),
            daemon=True,
        ).start()

    def _analysis_thread(self, input_str: str) -> None:
        stream = _QueueStream(self._q)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stream  # type: ignore[assignment]
        sys.stderr = stream  # type: ignore[assignment]
        try:
            csv_path = resolve_csv_input(Path(input_str))
            out_dir = run_analysis(csv_path)
            self._post_process_csvs(out_dir)
            self._q.put("\n[Done]\n")
            self.after(0, lambda: self._set_status("Done", "green"))
        except SystemExit as exc:
            code = exc.code
            self._q.put(f"\n[Exited with code {code}]\n")
            self.after(0, lambda c=code: self._set_status(f"Exited ({c})", "orange"))
        except Exception:
            self._q.put(f"\n[Error]\n{traceback.format_exc()}\n")
            self.after(0, lambda: self._set_status("Error", "red"))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.after(0, self._reset_run_btn)

    def _post_process_csvs(self, out_dir: Path) -> None:
        archive = self._archive_var.get()
        if archive:
            arch_dir = out_dir / "archive"
            arch_dir.mkdir(exist_ok=True)
        for name in _INTERMEDIATE_CSVS:
            src = out_dir / name
            if not src.is_file():
                continue
            if archive:
                src.rename(arch_dir / name)
                self._q.put(f"[Archive] Moved {name} → archive/\n")
            else:
                src.unlink()
                self._q.put(f"[Archive] Deleted {name}\n")

    def _set_status(self, text: str, color: str = "gray") -> None:
        self._status_label.configure(text=text, fg=color)

    def _reset_run_btn(self) -> None:
        self._running = False
        self._run_btn.configure(state="normal", text="Run Analysis")

    # ------------------------------------------------------------------ log

    def _poll_queue(self) -> None:
        try:
            while True:
                text = self._q.get_nowait()
                self._append_log(text)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _append_log(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
