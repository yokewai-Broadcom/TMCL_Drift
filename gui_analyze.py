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
        self.minsize(700, 560)
        self.resizable(True, True)

        self._q: queue.Queue[str] = queue.Queue()
        self._running = False
        self._status_label: tk.Label  # assigned in _build_ui

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
        ttk.Entry(path_frame, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", **pad
        )
        ttk.Button(path_frame, text="Browse…", width=8, command=self._browse_folder).grid(
            row=0, column=2, sticky="e", padx=8, pady=4
        )


        # ── run controls ────────────────────────────────────────────────────
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=12, pady=(4, 2))

        self._run_btn = ttk.Button(ctrl_frame, text="Run Analysis", command=self._on_run)
        self._run_btn.pack(side="left", padx=(0, 8))
        ttk.Button(ctrl_frame, text="Clear Log", command=self._clear_log).pack(side="left")

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

    # ------------------------------------------------------------------ browse

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self._input_var.set(path)


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
            run_analysis(csv_path)
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
