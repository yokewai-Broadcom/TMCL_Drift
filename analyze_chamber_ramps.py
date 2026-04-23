"""
Chamber 10–90% ramp analysis for ARU@100X.csv (-40 °C to 85 °C span).
Produces ramp_metrics.csv and timestamped figures_<YYYYMMDD_HHMMSS>/ per run.

CLI: pass a ``.csv`` path or a folder (a ``.csv`` under it is chosen; see ``resolve_csv_input``).

If a folder has exactly ``Initial.csv`` (case-insensitive) plus one other root ``.csv``,
ramps and overlay plots use the non-Initial file; soak boxplots prepend set 0 from the
first hot+cold soak in Initial.csv and add up to 100 cycles from the main file.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

SENTINEL = 9.9e37
T_COLD_REF = -40.0
T_HOT_REF = 85.0
DT_SPAN = T_HOT_REF - T_COLD_REF  # 125
T_10 = T_COLD_REF + 0.1 * DT_SPAN  # -27.5
T_90 = T_COLD_REF + 0.9 * DT_SPAN  # 72.5

D_COLS = [f"D{i}" for i in range(1, 16)]

_SKIP_DIR_CSV_NAMES = frozenset(
    {
        "ramp_metrics.csv",  # written by this tool; not chamber input
        "exclusion.csv",  # G×DUT omit list; not chamber data
    }
)

_EXCLUSION_TOKEN_RE = re.compile(r"^G(\d+)D(\d+)$", re.IGNORECASE)


def load_exclusion_set(path: Path | None) -> set[tuple[int, str]]:
    """
    Load group×DUT pairs to omit from soak resistance figures.
    Tokens: G1–G6 and D1–D15, e.g. G1D1, g4d13. Commas and newlines separate tokens;
    empty lines and # comments are ignored. Invalid tokens warn and are skipped.
    """
    if path is None or not path.is_file():
        return set()
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: set[tuple[int, str]] = set()
    for line in raw.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        for part in s.replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            m = _EXCLUSION_TOKEN_RE.match(part)
            if not m:
                print(f"Warning: exclusion.csv — skipping invalid token {part!r} in {path.name}", file=sys.stderr)
                continue
            g, dnum = int(m.group(1)), int(m.group(2))
            if not (1 <= g <= 6 and 1 <= dnum <= 15):
                print(
                    f"Warning: exclusion.csv — out of range G{g}D{dnum} (need G1–G6, D1–D15) in {path.name}",
                    file=sys.stderr,
                )
                continue
            out.add((g, f"D{dnum}"))
    return out


def read_data(csv_path: Path) -> pd.DataFrame:
    # index_col=False: otherwise pandas may treat the timestamp column as the index
    # and shift all fields (see misaligned Units/Group#).
    base = dict(skipinitialspace=True, index_col=False)

    def _read_c(**extra: object) -> pd.DataFrame:
        return pd.read_csv(csv_path, low_memory=False, **base, **extra)

    def _read_py(**extra: object) -> pd.DataFrame:
        # low_memory is invalid with engine="python" (pandas raises ValueError).
        return pd.read_csv(csv_path, engine="python", **base, **extra)

    try:
        df = _read_c()
    except UnicodeDecodeError:
        df = _read_c(encoding="latin-1")
    except pd.errors.ParserError:
        try:
            # pandas 2.2+: C engine can skip ragged rows (much faster than engine="python").
            df = _read_c(on_bad_lines="skip")
        except (TypeError, ValueError):
            try:
                try:
                    df = _read_py(on_bad_lines="skip")
                except TypeError:
                    # pandas < 1.3: no on_bad_lines
                    df = _read_py()
            except UnicodeDecodeError:
                try:
                    df = _read_py(encoding="latin-1", on_bad_lines="skip")
                except TypeError:
                    df = _read_py(encoding="latin-1")

    df.columns = df.columns.str.strip()
    time_col = "Units" if "Units" in df.columns else df.columns[0]
    # Logs may contain repeated CSV headers mid-file or ragged rows; keep only real timestamps.
    _ts_raw = df[time_col].astype(str).str.strip()
    _ts_ok = _ts_raw.str.match(
        r"^\d{1,2}/[A-Za-z]{3}/\d{4}\s+\d{1,2}:\d{2}:\d{2}$",
        na=False,
    )
    _n_bad = int((~_ts_ok).sum())
    if _n_bad:
        print(
            f"Dropping {_n_bad} row(s) in {csv_path.name}: "
            f"{time_col!r} is not a chamber timestamp (duplicate headers / corrupted lines)."
        )
        df = df.loc[_ts_ok].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df[time_col], format="%d/%b/%Y %H:%M:%S", dayfirst=True)
    for c in D_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] >= 1e36, c] = np.nan
    df["ChamberT(M)"] = pd.to_numeric(df["ChamberT(M)"], errors="coerce")
    return df


def chamber_series(df: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp; ChamberT is same for all groups."""
    g = df.groupby("timestamp", as_index=False).agg(
        T=("ChamberT(M)", "first"),
        ProgData=("ProgData", "first") if "ProgData" in df.columns else ("ChamberT(M)", "first"),
    )
    g = g.sort_values("timestamp").reset_index(drop=True)
    return g


def make_resistance_fn(df_: pd.DataFrame, ch_: pd.DataFrame):
    """Resistance samples in a chamber index window for a given raw dataframe."""

    def resistances_in_ch_index_range(
        i0: int,
        i1: int,
        group: str | None = None,
        dut_col: str | None = None,
    ) -> np.ndarray:
        t0 = ch_["timestamp"].iloc[i0]
        t1 = ch_["timestamp"].iloc[i1]
        mask = (df_["timestamp"] >= t0) & (df_["timestamp"] <= t1)
        if group is not None:
            mask = mask & (df_["Group#"].str.strip() == group.strip())
        sub = df_.loc[mask]
        if sub.empty:
            return np.array([])
        if dut_col is not None:
            if dut_col not in sub.columns:
                return np.array([])
            vals = sub[dut_col].to_numpy(dtype=float)
        else:
            vals = sub[D_COLS].to_numpy(dtype=float).ravel()
        return vals[np.isfinite(vals)]

    return resistances_in_ch_index_range


def label_plateau(T: np.ndarray, cold_max: float, hot_min: float) -> np.ndarray:
    """0=cold soak, 1=hot soak, -1=neither (ramp or startup)."""
    out = np.full(len(T), -1, dtype=np.int8)
    out[T <= cold_max] = 0
    out[T >= hot_min] = 1
    return out


def contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """List of (start, end inclusive, label value)."""
    if len(labels) == 0:
        return []
    runs = []
    s = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            runs.append((s, i - 1, int(cur)))
            s = i
            cur = labels[i]
    runs.append((s, len(labels) - 1, int(cur)))
    return runs


def interpolate_crossing(
    t: np.ndarray,
    T: np.ndarray,
    thresh: float,
    direction: str,
) -> float | None:
    """
    direction: 'up' — first crossing where T goes from below to at/above thresh.
               'down' — first crossing where T goes from above to at/below thresh.
    """
    for i in range(len(T) - 1):
        T0, T1 = T[i], T[i + 1]
        t0, t1 = t[i], t[i + 1]
        if direction == "up":
            if T0 < thresh <= T1:
                if T1 == T0:
                    return float(t0)
                frac = (thresh - T0) / (T1 - T0)
                return float(t0 + frac * (t1 - t0))
        else:
            if T0 > thresh >= T1:
                if T1 == T0:
                    return float(t0)
                frac = (thresh - T0) / (T1 - T0)
                return float(t0 + frac * (t1 - t0))
    return None


def find_trimmed_cold_soak_intervals(
    T: np.ndarray,
    *,
    threshold: float = -40.0,
    n_edge: int = 5,
) -> list[tuple[int, int]]:
    """
    Cold soak resistance window (indices into T, inclusive):
    - Ramp down: first crossing with T[i] > threshold and T[i+1] <= threshold.
    - Require n_edge consecutive readings with T <= threshold (stabilization).
    - Data starts at index (streak_start + n_edge).
    - Ramp up: first j with T[j-1] <= threshold and T[j] > threshold.
    - Exclude last n_edge readings before that up-cross (last kept index is j-1-n_edge).
    """
    n = len(T)
    out: list[tuple[int, int]] = []
    search_from = 0
    while search_from < n - 1:
        i = None
        for k in range(search_from, n - 1):
            if T[k] > threshold and T[k + 1] <= threshold:
                i = k
                break
        if i is None:
            break
        streak_start = None
        s0 = i + 1
        for s in range(s0, n - n_edge + 1):
            if bool(np.all(T[s : s + n_edge] <= threshold)):
                streak_start = s
                break
        if streak_start is None:
            search_from = i + 1
            continue
        data_start = streak_start + n_edge
        v = None
        for j in range(max(data_start + 1, i + 2), n):
            if T[j - 1] <= threshold and T[j] > threshold:
                v = j
                break
        if v is None:
            break
        data_end = v - 1 - n_edge
        if data_end >= data_start:
            out.append((data_start, data_end))
        search_from = v
    return out


def find_trimmed_hot_soak_intervals(
    T: np.ndarray,
    *,
    threshold: float = 85.0,
    n_edge: int = 5,
) -> list[tuple[int, int]]:
    """
    Hot soak resistance window (indices inclusive):
    - Ramp up beyond threshold: T[i] <= threshold and T[i+1] > threshold.
    - Require n_edge consecutive readings with T > threshold.
    - Data starts at streak_start + n_edge.
    - Ramp down below threshold: T[w-1] > threshold and T[w] <= threshold.
    - Exclude last n_edge readings before down-cross (end = w-1-n_edge).
    """
    n = len(T)
    out: list[tuple[int, int]] = []
    search_from = 0
    while search_from < n - 1:
        i = None
        for k in range(search_from, n - 1):
            if T[k] <= threshold and T[k + 1] > threshold:
                i = k
                break
        if i is None:
            break
        streak_start = None
        s0 = i + 1
        for s in range(s0, n - n_edge + 1):
            if bool(np.all(T[s : s + n_edge] > threshold)):
                streak_start = s
                break
        if streak_start is None:
            search_from = i + 1
            continue
        data_start = streak_start + n_edge
        w = None
        for j in range(max(data_start + 1, i + 2), n):
            if T[j - 1] > threshold and T[j] <= threshold:
                w = j
                break
        if w is None:
            break
        data_end = w - 1 - n_edge
        if data_end >= data_start:
            out.append((data_start, data_end))
        search_from = w
    return out


def find_ramps(ch: pd.DataFrame, cold_max: float, hot_min: float) -> tuple[list[dict], list[dict]]:
    """
    Return lists of heating and cooling ramp dicts with slice indices into ch.
    Each ramp is bounded by plateau labels (cold then hot, or hot then cold).
    """
    T = ch["T"].to_numpy(dtype=float)
    t_sec = (ch["timestamp"] - ch["timestamp"].iloc[0]).dt.total_seconds().to_numpy()

    labels = label_plateau(T, cold_max, hot_min)
    runs = contiguous_runs(labels)

    heating: list[dict] = []
    cooling: list[dict] = []

    for j in range(len(runs)):
        s0, e0, lab0 = runs[j]
        if lab0 != 0:
            continue
        # cold plateau: find next hot plateau
        k = j + 1
        while k < len(runs) and runs[k][2] != 1:
            k += 1
        if k >= len(runs) or runs[k][2] != 1:
            continue
        hot_start = runs[k][0]
        # Include last cold-soak point (T <= cold_max) so an upward −27.5 °C
        # crossing is not missed when the first post-cold sample jumps above it.
        start_idx = e0
        # Include first hot-soak sample so 72.5 °C may be crossed between the
        # last ramp point and the first plateau sample.
        end_idx = hot_start
        if start_idx <= end_idx:
            heating.append(
                {
                    "i0": start_idx,
                    "i1": end_idx,
                    "t_sec": t_sec,
                    "T": T,
                }
            )

    for j in range(len(runs)):
        s0, e0, lab0 = runs[j]
        if lab0 != 1:
            continue
        k = j + 1
        while k < len(runs) and runs[k][2] != 0:
            k += 1
        if k >= len(runs) or runs[k][2] != 0:
            continue
        cold_start = runs[k][0]
        # Include last hot-soak point for the same reason as heating (72.5 °C down).
        start_idx = e0
        end_idx = cold_start
        if start_idx <= end_idx:
            cooling.append(
                {
                    "i0": start_idx,
                    "i1": end_idx,
                    "t_sec": t_sec,
                    "T": T,
                }
            )

    return heating, cooling


def measure_heating_10_90(ramp: dict) -> tuple[float | None, float | None, str]:
    i0, i1 = ramp["i0"], ramp["i1"]
    t_sec = ramp["t_sec"][i0 : i1 + 1]
    T = ramp["T"][i0 : i1 + 1]
    if len(T) < 2:
        return None, None, "too_few_points"
    # require net heating over window
    if T[-1] <= T[0]:
        return None, None, "not_net_heating"
    t10 = interpolate_crossing(t_sec, T, T_10, "up")
    t90 = interpolate_crossing(t_sec, T, T_90, "up")
    if t10 is None or t90 is None:
        return None, None, "missing_crossing"
    if t90 < t10:
        return None, None, "inverted_crossings"
    return t10, t90, "ok"


def ramp_endpoint_rate_c_per_s(ramp: dict) -> float:
    """Average T rate over ramp window (first to last sample in segment), in °C/s (x60 for °C/min)."""
    i0, i1 = ramp["i0"], ramp["i1"]
    t_sec = ramp["t_sec"]
    T = ramp["T"]
    dt = float(t_sec[i1] - t_sec[i0])
    if dt <= 0:
        return float("nan")
    return float((T[i1] - T[i0]) / dt)


def measure_cooling_10_90(ramp: dict) -> tuple[float | None, float | None, str]:
    i0, i1 = ramp["i0"], ramp["i1"]
    t_sec = ramp["t_sec"][i0 : i1 + 1]
    T = ramp["T"][i0 : i1 + 1]
    if len(T) < 2:
        return None, None, "too_few_points"
    if T[-1] >= T[0]:
        return None, None, "not_net_cooling"
    # cooling: high to low — first down crossing of 72.5 then -27.5
    t_hi = interpolate_crossing(t_sec, T, T_90, "down")
    t_lo = interpolate_crossing(t_sec, T, T_10, "down")
    if t_hi is None or t_lo is None:
        return None, None, "missing_crossing"
    if t_lo < t_hi:
        return None, None, "inverted_crossings"
    return t_hi, t_lo, "ok"


def cooling_ramp_10_90_rate_c_per_min(ramp: dict) -> float:
    """Signed mean rate over T_90→T_10 (100 C) on 10–90% window, °C/min (negative = cooling)."""
    t_hi, t_lo, st = measure_cooling_10_90(ramp)
    if st != "ok" or t_hi is None or t_lo is None:
        return float("nan")
    dt = float(t_lo - t_hi)
    if dt <= 0:
        return float("nan")
    return float((T_10 - T_90) / dt * 60.0)


def _app_base_dir() -> Path:
    """Directory containing the script, or the folder with the frozen .exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _root_csv_candidates(folder: Path) -> list[Path]:
    """Non-skipped ``*.csv`` files in ``folder`` root only (sorted by name)."""
    folder = folder.resolve()
    out: list[Path] = []
    for p in sorted(folder.glob("*.csv"), key=lambda x: x.name.lower()):
        if p.name.lower() in _SKIP_DIR_CSV_NAMES:
            continue
        out.append(p)
    return out


def _resolve_csv_for_analysis(csv_path: Path) -> tuple[Path, bool, Path | None]:
    """
    If ``csv_path`` sits in a folder with exactly ``Initial.csv`` (case-insensitive)
    plus one other root CSV, enable dual soak mode and ensure the returned path is
    the **main** chamber file (never Initial) for ramps/metrics/overlays.

    If Initial.csv is present and there are more than two root CSVs, exit with an error.
    """
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    roots = _root_csv_candidates(csv_path.parent)
    init = next((p for p in roots if p.name.lower() == "initial.csv"), None)
    if init is None:
        return csv_path, False, None
    if len(roots) > 2:
        print(
            f"Error: {csv_path.parent} contains Initial.csv with {len(roots)} CSV files "
            f"({', '.join(p.name for p in roots)}). Use only Initial.csv plus one main "
            "chamber log, or pass the main file explicitly with --csv.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(roots) < 2:
        return csv_path, False, None
    mains = [p for p in roots if p.name.lower() != "initial.csv"]
    if len(mains) != 1:
        return csv_path, False, None
    main_p = mains[0]
    if csv_path.resolve() == init.resolve():
        return main_p, True, init
    return csv_path, True, init


def _annotate_soak_row_global_stats(ax: plt.Axes, series_list: list[np.ndarray]) -> None:
    """
    Whiskers aligned with matplotlib boxplot (whis=1.5, no outliers drawn): use each
    series' whisker hi/lo from boxplot_stats; max/min marker y = max(whishi), min(whislo)
    across DUTs. A middle marker at pooled mean (between whisker hi/lo). Percentages vs
    pooled mean only beside max/min; markers drawn outside data area (axes x>1).
    Each label shows the value line below the % or "middle" line.
    """
    if not series_list:
        return
    parts = [np.asarray(a, dtype=float)[np.isfinite(a)] for a in series_list if a.size > 0]
    if not parts:
        return
    vals = np.concatenate(parts)
    if vals.size == 0:
        return
    vmean = float(np.mean(vals))

    try:
        stats = boxplot_stats(parts, whis=1.5)
    except (ValueError, ZeroDivisionError):
        return
    if not stats:
        return
    whishi = max(float(s["whishi"]) for s in stats)
    whislo = min(float(s["whislo"]) for s in stats)

    pct_up = None if vmean == 0 else 100.0 * (whishi - vmean) / vmean
    pct_dn = None if whislo == 0 else 100.0 * (vmean - whislo) / whislo

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    x0, x1 = 1.01, 1.10
    ax.plot([x0, x1], [whishi, whishi], transform=trans, color="0.2", linewidth=1.5, clip_on=False)
    ax.plot(
        [x0, x1],
        [vmean, vmean],
        transform=trans,
        color="0.35",
        linewidth=1.2,
        linestyle="--",
        clip_on=False,
    )
    ax.plot([x0, x1], [whislo, whislo], transform=trans, color="0.2", linewidth=1.5, linestyle=":", clip_on=False)

    def _fmt_r(v: float) -> str:
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.6g}"

    pct_hi = f"{pct_up:.2f}%" if pct_up is not None else "N/A"
    pct_lo = f"{pct_dn:.2f}%" if pct_dn is not None else "N/A"
    ax.text(
        1.12,
        whishi,
        f"  {pct_hi}\n  {_fmt_r(whishi)}",
        transform=trans,
        ha="left",
        va="center",
        fontsize=7,
        color="0.2",
        linespacing=0.9,
        clip_on=False,
    )
    ax.text(
        1.12,
        vmean,
        f"  middle\n  {_fmt_r(vmean)}",
        transform=trans,
        ha="left",
        va="center",
        fontsize=7,
        color="0.35",
        linespacing=0.9,
        clip_on=False,
    )
    ax.text(
        1.12,
        whislo,
        f"  {pct_lo}\n  {_fmt_r(whislo)}",
        transform=trans,
        ha="left",
        va="center",
        fontsize=7,
        color="0.2",
        linespacing=0.9,
        clip_on=False,
    )


def run_analysis(
    csv_path: Path,
    *,
    out_dir: Path | None = None,
    cold_max: float = -32.0,
    hot_min: float = 78.0,
    soak_edge_readings: int = 5,
    soak_cold_th: float = -40.0,
    soak_hot_th: float = 85.0,
    exclusion_csv: Path | None = None,
) -> Path:
    """
    Run chamber ramp analysis on ``csv_path``.

    Writes ``ramp_metrics.csv`` under ``out_dir`` and PNG figures in a new subfolder
    ``figures_<YYYYMMDD_HHMMSS>`` each run.

    Soak resistance figures respect optional ``exclusion_csv`` (default:
    ``csv_path.parent / "exclusion.csv"``) listing ``G*D*`` pairs to omit.

    Returns the resolved output directory.
    """
    csv_path, dual_soak, initial_csv_path = _resolve_csv_for_analysis(csv_path.resolve())

    out_dir_resolved = (out_dir if out_dir is not None else csv_path.parent).resolve()
    fig_dir = out_dir_resolved / f"figures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures directory: {fig_dir}")

    if dual_soak and initial_csv_path is not None:
        print(
            "Dual CSV: ramps/overlays/metrics use main file; soak boxplots = "
            "set 0 (first hot+cold soak in Initial.csv) + up to 100 cycles from main."
        )

    df = read_data(csv_path)
    ch = chamber_series(df)
    print(f"Loaded main {csv_path.name}: {len(df)} rows, {len(ch)} unique timestamps.")

    heating, cooling = find_ramps(ch, cold_max, hot_min)
    print(f"Detected {len(heating)} heating ramps, {len(cooling)} cooling ramps.")
    print(f"Thresholds: T10={T_10} C, T90={T_90} C (span {T_COLD_REF} to {T_HOT_REF} C).")

    rows = []
    for idx, ramp in enumerate(heating):
        t10, t90, status = measure_heating_10_90(ramp)
        dt = (t90 - t10) if (t10 is not None and t90 is not None) else np.nan
        i0, i1 = ramp["i0"], ramp["i1"]
        rows.append(
            {
                "ramp_type": "heating",
                "ramp_index": idx,
                "t_start": ch["timestamp"].iloc[i0],
                "t_end": ch["timestamp"].iloc[i1],
                "t10_cross_s": t10,
                "t90_cross_s": t90,
                "rise_time_10_90_s": dt,
                "fall_time_10_90_s": np.nan,
                "time_10_90_s": dt,
                "status": status,
            }
        )

    for idx, ramp in enumerate(cooling):
        t_hi, t_lo, status = measure_cooling_10_90(ramp)
        dt = (t_lo - t_hi) if (t_hi is not None and t_lo is not None) else np.nan
        i0, i1 = ramp["i0"], ramp["i1"]
        rows.append(
            {
                "ramp_type": "cooling",
                "ramp_index": idx,
                "t_start": ch["timestamp"].iloc[i0],
                "t_end": ch["timestamp"].iloc[i1],
                "t10_cross_s": t_hi,
                "t90_cross_s": t_lo,
                "rise_time_10_90_s": np.nan,
                "fall_time_10_90_s": dt,
                "time_10_90_s": dt,
                "status": status,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir_resolved / "ramp_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    # Overlay ChamberT vs time-from-ramp-start (seconds from i0)
    heat_rates = np.array([ramp_endpoint_rate_c_per_s(r) for r in heating], dtype=float)
    n_hr = int(np.sum(np.isfinite(heat_rates)))
    c_per_min = 60.0
    avg_ramp_up = float(np.nanmean(heat_rates)) * c_per_min

    cool_1090_c_per_min = np.array(
        [cooling_ramp_10_90_rate_c_per_min(r) for r in cooling], dtype=float
    )
    n_cr = int(np.sum(np.isfinite(cool_1090_c_per_min)))
    avg_ramp_down = float(np.nanmean(cool_1090_c_per_min))
    print(
        f"Mean ramp-up rate (endpoint): {avg_ramp_up:.4f} C/min over {n_hr} heating ramps; "
        f"mean ramp-down (10-90% T span, n={n_cr}): {avg_ramp_down:.4f} C/min."
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ramp in heating:
        i0, i1 = ramp["i0"], ramp["i1"]
        tt = ramp["t_sec"][i0 : i1 + 1] - ramp["t_sec"][i0]
        TT = ramp["T"][i0 : i1 + 1]
        axes[0].plot(tt, TT, color="C0", alpha=0.12, linewidth=1)
    axes[0].axhline(T_10, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].axhline(T_90, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Time since ramp start (s)")
    axes[0].set_ylabel("ChamberT(M) (°C)")
    axes[0].set_title("Heating ramps (overlay)")
    axes[0].set_ylim(-45, 95)
    axes[0].text(
        0.03,
        0.97,
        f"Avg ramp-up rate:\n{avg_ramp_up:.3f} C/min\n(n={n_hr})",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.85, edgecolor="0.5"),
    )

    # Profile cooling ramps only (no synthetic 85->-40 combined curve on overlay).
    for ramp in cooling:
        i0, i1 = ramp["i0"], ramp["i1"]
        tt = ramp["t_sec"][i0 : i1 + 1] - ramp["t_sec"][i0]
        TT = ramp["T"][i0 : i1 + 1]
        axes[1].plot(tt, TT, color="C1", alpha=0.12, linewidth=1)
    axes[1].axhline(T_10, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[1].axhline(T_90, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Time since ramp start (s)")
    axes[1].set_title("Cooling ramps (overlay)")
    axes[1].text(
        0.03,
        0.97,
        f"Avg ramp-down rate:\n{avg_ramp_down:.3f} C/min\n(n={n_cr})",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="lightcyan", alpha=0.85, edgecolor="0.5"),
    )
    fig.suptitle("ChamberT(M): all ramps (raw)", y=1.02)
    fig.tight_layout()
    oc_path = fig_dir / "overlay_chamber_ramps.png"
    fig.savefig(oc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {oc_path}")

    exclusion_path = (csv_path.parent / "exclusion.csv") if exclusion_csv is None else exclusion_csv
    excluded = load_exclusion_set(exclusion_path.resolve())
    if excluded:
        print(
            f"Soak figures: omitting {len(excluded)} group×DUT pair(s) from {exclusion_path.name} "
            f"({', '.join(f'G{g}{d}' for g, d in sorted(excluded))})"
        )
    else:
        print(f"Soak figures: no group×DUT exclusions (read {exclusion_path.name}).")

    # --- Soak resistance (trimmed): 5 readings after crossing in, drop 5 before crossing out ---
    T_arr = ch["T"].to_numpy(dtype=float)
    ne = soak_edge_readings
    cold_main = find_trimmed_cold_soak_intervals(
        T_arr, threshold=soak_cold_th, n_edge=ne
    )
    hot_main = find_trimmed_hot_soak_intervals(
        T_arr, threshold=soak_hot_th, n_edge=ne
    )
    n_cold_seg, n_hot_seg = len(cold_main), len(hot_main)
    n_plot_main = min(n_cold_seg, n_hot_seg, 100)
    if n_cold_seg != n_hot_seg or n_plot_main < 100:
        print(
            f"Soak segments (main): cold={n_cold_seg}, hot={n_hot_seg}; "
            f"using first {n_plot_main} paired cycles from main file."
        )
    cold_main = cold_main[:n_plot_main]
    hot_main = hot_main[:n_plot_main]

    res_main = make_resistance_fn(df, ch)
    use_init0 = False
    hot_tagged: list[tuple[str, tuple[int, int]]] = [("main", iv) for iv in hot_main]
    cold_tagged: list[tuple[str, tuple[int, int]]] = [("main", iv) for iv in cold_main]

    if dual_soak and initial_csv_path is not None:
        df_init = read_data(initial_csv_path)
        ch_init = chamber_series(df_init)
        print(
            f"Loaded Initial.csv: {len(df_init)} rows, {len(ch_init)} unique timestamps "
            "(soak set 0 only)."
        )
        T_init = ch_init["T"].to_numpy(dtype=float)
        cold_init = find_trimmed_cold_soak_intervals(
            T_init, threshold=soak_cold_th, n_edge=ne
        )
        hot_init = find_trimmed_hot_soak_intervals(
            T_init, threshold=soak_hot_th, n_edge=ne
        )
        if len(hot_init) > 0 and len(cold_init) > 0:
            use_init0 = True
            res_init = make_resistance_fn(df_init, ch_init)
            hot_tagged = [("init", hot_init[0])] + hot_tagged
            cold_tagged = [("init", cold_init[0])] + cold_tagged
        else:
            print(
                "Initial.csv: missing first hot or cold soak segment; "
                "skipping set 0 (main cycles only)."
            )
            res_init = res_main  # unused
    else:
        res_init = res_main  # unused

    def resistances_tagged(
        src: str,
        i0: int,
        i1: int,
        group: str | None,
        dut_col: str | None,
    ) -> np.ndarray:
        fn = res_init if src == "init" else res_main
        return fn(i0, i1, group, dut_col=dut_col)

    n_cyc = len(hot_tagged)
    if len(cold_tagged) != n_cyc:
        n_cyc = min(len(hot_tagged), len(cold_tagged))
        hot_tagged = hot_tagged[:n_cyc]
        cold_tagged = cold_tagged[:n_cyc]

    cycle_index_base = 0 if (dual_soak and use_init0) else 1
    cycle_xlabel = (
        "Cycle index (0 = first soak in Initial.csv; 1+ = main log)"
        if (dual_soak and use_init0)
        else "Cycle index (1 = first paired soak segment in log)"
    )

    def save_soak_boxplot_hot_cold_combined(
        hot_per_cycle: list[np.ndarray],
        cold_per_cycle: list[np.ndarray],
        *,
        title: str,
        fname: str,
        ylabel: str,
        cycle_index_base: int = 1,
        xlabel_cycle: str | None = None,
    ) -> None:
        """Two stacked panels: n hot dwell boxplots (top), n cold dwell boxplots (bottom)."""
        n = min(len(hot_per_cycle), len(cold_per_cycle))
        if n == 0:
            return
        hot_use = hot_per_cycle[:n]
        cold_use = cold_per_cycle[:n]
        fig_w = max(18.0, 0.11 * n)
        fig, (ax_hot, ax_cold) = plt.subplots(
            2,
            1,
            figsize=(fig_w, 8.0),
            sharex=True,
            sharey=False,
            gridspec_kw={"hspace": 0.28},
        )
        if cycle_index_base == 0:
            positions = np.arange(0, n, dtype=float)
        else:
            positions = np.arange(1, n + 1, dtype=float)
        w = min(0.55, 75.0 / max(1, n))
        edge = "0.35"

        def _row(ax, series: list[np.ndarray], face: str, row_title: str) -> None:
            bp = ax.boxplot(
                series,
                positions=positions,
                widths=w,
                showfliers=False,
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(face)
                patch.set_edgecolor(edge)
            for key in ("whiskers", "caps"):
                for line in bp[key]:
                    line.set_color(edge)
            for m in bp["medians"]:
                m.set_color("C0")
                m.set_linewidth(1.5)
            ax.set_title(row_title, loc="left", fontsize=10)
            step = max(1, n // 25)
            xt = positions[::step]
            ax.set_xticks(xt)
            ax.set_xticklabels([str(int(x)) for x in xt], rotation=45, ha="right", fontsize=7)
            ax.margins(x=0.005)

        _row(
            ax_hot,
            hot_use,
            "#ffb399",
            f"Hot dwell (T > {soak_hot_th:g} °C trimmed), n={n} cycles",
        )
        ax_hot.tick_params(axis="x", labelbottom=False)
        ax_hot.set_ylabel(ylabel)
        _row(
            ax_cold,
            cold_use,
            "#9ecae9",
            f"Cold dwell (T ≤ {soak_cold_th:g} °C trimmed), n={n} cycles",
        )
        ax_cold.set_ylabel(ylabel)
        ax_cold.set_xlabel(
            xlabel_cycle if xlabel_cycle is not None else "Cycle index (1 = first paired soak segment in log)"
        )
        fig.suptitle(title, fontsize=9, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        outp = fig_dir / fname
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outp}")

    soak_short = (
        f"Hot: T>{soak_hot_th} C; cold: T<={soak_cold_th} C; "
        f"+{ne} readings after crossing in, -{ne} before crossing out"
    )
    soak_pool_title_suffix = (
        " (Initial set 0 + main cycles)"
        if (dual_soak and use_init0)
        else ""
    )

    for gi in range(1, 7):
        gname = f"Grp {gi}"
        gtag = gname.replace(" ", "")

        # One figure: hot (top) + cold (bottom), D1–D15 on shared x (slot per DUT), cycles pooled.
        def _merged_soak_tagged(
            intervals: list[tuple[str, tuple[int, int]]], dut: str
        ) -> np.ndarray:
            parts = [
                resistances_tagged(src, i0, i1, gname, dut_col=dut)
                for src, (i0, i1) in intervals
            ]
            nonempty = [p for p in parts if p.size > 0]
            return np.concatenate(nonempty) if nonempty else np.array([])

        hot_series: list[np.ndarray] = []
        hot_pos: list[int] = []
        cold_series: list[np.ndarray] = []
        cold_pos: list[int] = []
        for slot, d in enumerate(D_COLS, start=1):
            if d not in df.columns:
                continue
            if (gi, d) in excluded:
                continue
            h_m = _merged_soak_tagged(hot_tagged, d)
            if h_m.size > 0:
                hot_series.append(h_m)
                hot_pos.append(slot)
            c_m = _merged_soak_tagged(cold_tagged, d)
            if c_m.size > 0:
                cold_series.append(c_m)
                cold_pos.append(slot)

        if hot_series or cold_series:
            fig_w = max(12.0, 0.72 * len(D_COLS))
            fig, (ax_h, ax_c) = plt.subplots(
                2,
                1,
                figsize=(fig_w, 8.0),
                sharex=True,
                sharey=False,
                gridspec_kw={"hspace": 0.28},
            )
            x_ticks = np.arange(1, len(D_COLS) + 1, dtype=float)
            x_labs = list(D_COLS)
            edge = "0.35"
            w = 0.55

            def _dut_row(ax, series: list[np.ndarray], pos: list[int], face: str, row_title: str) -> None:
                if not series:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    return
                bp = ax.boxplot(
                    series,
                    positions=pos,
                    widths=w,
                    showfliers=False,
                    patch_artist=True,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(face)
                    patch.set_edgecolor(edge)
                for key in ("whiskers", "caps"):
                    for line in bp[key]:
                        line.set_color(edge)
                for m in bp["medians"]:
                    m.set_color("C0")
                    m.set_linewidth(1.5)
                ax.set_title(row_title, loc="left", fontsize=10)

            _dut_row(
                ax_h,
                hot_series,
                hot_pos,
                "#ffb399",
                (
                    f"Hot dwell (T > {soak_hot_th:g} °C), {gname} — by DUT "
                    f"(pooled n={n_cyc} cycles){soak_pool_title_suffix}"
                ),
            )
            ax_h.set_ylabel(f"Resistance ({gname})")
            ax_h.tick_params(axis="x", labelbottom=False)
            _dut_row(
                ax_c,
                cold_series,
                cold_pos,
                "#9ecae9",
                (
                    f"Cold dwell (T ≤ {soak_cold_th:g} °C), {gname} — by DUT "
                    f"(pooled n={n_cyc} cycles){soak_pool_title_suffix}"
                ),
            )
            ax_c.set_ylabel(f"Resistance ({gname})")

            for ax in (ax_h, ax_c):
                ax.set_xlim(0.5, len(D_COLS) + 0.5)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labs, rotation=45, ha="right", fontsize=8)
                ax.margins(x=0.01)
            if hot_series:
                _annotate_soak_row_global_stats(ax_h, hot_series)
            if cold_series:
                _annotate_soak_row_global_stats(ax_c, cold_series)
            ax_c.set_xlabel("DUT")
            fig.suptitle(
                f"Soak R (trimmed) — {gname}, D1–D15 (hot top / cold bottom){soak_pool_title_suffix}\n"
                f"{soak_short}",
                fontsize=9,
                y=0.995,
            )
            # Manual margins: tight_layout warns when markers use blended transform (axes x>1).
            fig.subplots_adjust(left=0.07, right=0.84, bottom=0.09, top=0.90, hspace=0.28)
            outp = fig_dir / f"boxplot_soak_resistance_{gtag}_D1toD15_hot_cold.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {outp}")

        # Per group × per DUT: one PNG (e.g. boxplot_soak_G1_D1_hot_cold_100cycl.png); skip if excluded.
        g_short = f"G{gi}"
        for d in D_COLS:
            if d not in df.columns:
                continue
            if (gi, d) in excluded:
                continue
            hot_gd = [
                resistances_tagged(src, hs, he, gname, dut_col=d)
                for src, (hs, he) in hot_tagged
            ]
            cold_gd = [
                resistances_tagged(src, cs0, cs1, gname, dut_col=d)
                for src, (cs0, cs1) in cold_tagged
            ]
            save_soak_boxplot_hot_cold_combined(
                hot_gd,
                cold_gd,
                title=(
                    f"Soak R (trimmed) — {g_short}-{d}: {len(hot_gd)} hot + "
                    f"{len(cold_gd)} cold dwells per cycle{soak_pool_title_suffix}\n{soak_short}"
                ),
                fname=f"boxplot_soak_{g_short}_{d}_hot_cold_{n_cyc}cycl.png",
                ylabel=f"Resistance ({g_short}, {d})",
                cycle_index_base=cycle_index_base,
                xlabel_cycle=cycle_xlabel,
            )

    return out_dir_resolved


def _is_figures_output_dirname(name: str) -> bool:
    """True for legacy ``figures`` or timestamped ``figures_%Y%m%d_%H%M%S`` output folders."""
    n = name.lower()
    return n == "figures" or (n.startswith("figures_") and len(n) > len("figures_"))


def _csv_candidates_in_folder(folder: Path) -> list[Path]:
    """CSV paths under ``folder``: root ``*.csv`` first, else recursive (skip figures output dirs, skipped names)."""
    folder = folder.resolve()

    def _skip_path(p: Path) -> bool:
        if p.name.lower() in _SKIP_DIR_CSV_NAMES:
            return True
        rel = p.relative_to(folder)
        if any(_is_figures_output_dirname(part) for part in rel.parts[:-1]):
            return True
        return False

    seen: set[Path] = set()
    out: list[Path] = []

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp in seen or _skip_path(p):
            return
        seen.add(rp)
        out.append(p)

    for p in sorted(folder.glob("*.csv"), key=lambda x: x.name.lower()):
        _add(p)
    if not out:
        for p in sorted(folder.rglob("*.csv"), key=lambda x: str(x).lower()):
            _add(p)
    return sorted(out, key=lambda p: p.name.lower())


def _pick_csv_from_dir(folder: Path) -> Path:
    """Pick one CSV under ``folder`` for analysis."""
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    roots = _root_csv_candidates(folder)
    if len(roots) == 2 and any(p.name.lower() == "initial.csv" for p in roots):
        return next(p for p in roots if p.name.lower() != "initial.csv")
    csvs = _csv_candidates_in_folder(folder)
    if not csvs:
        raise FileNotFoundError(
            f"No suitable .csv under {folder} (skipped {_SKIP_DIR_CSV_NAMES}; "
            "ignored CSVs under figures/ or figures_<timestamp>/ subfolders)."
        )
    if len(csvs) > 1:
        preview = ", ".join(p.name for p in csvs[:8])
        if len(csvs) > 8:
            preview += f", … ({len(csvs)} total)"
        print(
            f"Warning: multiple CSV files in {folder}; using {csvs[0].name}\n"
            f"  Pass a specific file with --csv if needed. Found: {preview}",
            file=sys.stderr,
        )
    return csvs[0]


def resolve_csv_input(path: Path) -> Path:
    """
    Accept a path to a ``.csv`` file or to a folder containing exactly one (or a chosen) CSV.
    """
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            print(f"Error: input file must be .csv: {path}", file=sys.stderr)
            sys.exit(1)
        return path
    if path.is_dir():
        try:
            return _pick_csv_from_dir(path)
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"Error: path not found (not a file or folder): {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Chamber ramp analysis. Pass a CSV file, a folder containing a .csv, "
            "or drag a CSV onto the .exe."
        )
    )
    p.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=None,
        metavar="CSV_OR_FOLDER",
        help="Input .csv file, or folder that contains the .csv to analyze.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input .csv file or folder (overrides positional when set).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for ramp_metrics.csv and figures_<timestamp>/ (default: same folder as the input CSV).",
    )
    p.add_argument(
        "--exclusion-csv",
        type=Path,
        default=None,
        help="Optional G×DUT omit list (default: exclusion.csv next to the input CSV).",
    )
    p.add_argument("--cold-max", type=float, default=-32.0, help="T <= this = cold soak")
    p.add_argument("--hot-min", type=float, default=78.0, help="T >= this = hot soak")
    p.add_argument(
        "--soak-edge-readings",
        type=int,
        default=5,
        help="Consecutive T in-soak readings to wait before R; also trim this many before leaving soak",
    )
    p.add_argument("--soak-cold-th", type=float, default=-40.0)
    p.add_argument("--soak-hot-th", type=float, default=85.0)
    args = p.parse_args()

    csv_path = args.csv if args.csv is not None else args.csv_file
    if csv_path is None:
        csv_path = _app_base_dir() / "ARU@100X.csv"
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            print(f"Error: default CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        csv_path = resolve_csv_input(csv_path)

    run_analysis(
        csv_path,
        out_dir=args.out_dir,
        cold_max=args.cold_max,
        hot_min=args.hot_min,
        soak_edge_readings=args.soak_edge_readings,
        soak_cold_th=args.soak_cold_th,
        soak_hot_th=args.soak_hot_th,
        exclusion_csv=args.exclusion_csv,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
