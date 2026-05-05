"""
report_generator.py
===================
Load result CSVs, generate LaTeX tables, and copy figures into
the report/ directory so the final PDF can be compiled in one step.

Usage
-----
    python src/report_generator.py
"""

import os
import sys
import shutil

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------

def _df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_fmt: str = "{:.4f}",
) -> str:
    """
    Convert a DataFrame to a LaTeX table string (booktabs style).

    Parameters
    ----------
    df : pd.DataFrame
    caption : str
        Table caption.
    label : str
        LaTeX \\label{...} key.
    float_fmt : str
        Python format string for float values.

    Returns
    -------
    str
        Complete LaTeX table environment.
    """
    col_spec = "l" + "r" * len(df.columns)
    header   = " & ".join([""] + list(df.columns)) + r" \\"

    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for val in row:
            if isinstance(val, float):
                cells.append(float_fmt.format(val))
            else:
                cells.append(str(val))
        rows.append(" & ".join(cells) + r" \\")

    body = "\n    ".join(rows)

    table = (
        r"\begin{table}[H]" "\n"
        r"    \centering" "\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        r"    \begin{tabular}{" + col_spec + "}\n"
        r"        \toprule" "\n"
        f"        {header}\n"
        r"        \midrule" "\n"
        f"        {body}\n"
        r"        \bottomrule" "\n"
        r"    \end{tabular}" "\n"
        r"\end{table}"
    )
    return table


def generate_latency_table(latency_csv: str = None) -> str:
    """
    Load results/tables/latency_summary.csv and write
    report/tables/latency_table.tex.

    Returns
    -------
    str
        LaTeX table string.
    """
    latency_csv = latency_csv or os.path.join(
        config.RESULTS_TABLES, "latency_summary.csv"
    )
    _check_file(latency_csv)

    df = pd.read_csv(latency_csv)
    cols = ["method", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "p95_ms"]
    df   = df[[c for c in cols if c in df.columns]]
    df   = df.rename(columns={
        "method":     "Method",
        "mean_ms":    "Mean (ms)",
        "median_ms":  "Median (ms)",
        "std_ms":     "Std (ms)",
        "min_ms":     "Min (ms)",
        "max_ms":     "Max (ms)",
        "p95_ms":     "P95 (ms)",
    }).set_index("Method")

    latex = _df_to_latex(
        df,
        caption="Latency statistics per VaR computation method.",
        label="tab:latency",
    )
    _save_tex(latex, "latency_table.tex")
    return latex


def generate_backtest_table(backtest_csv: str = None) -> str:
    """
    Load results/tables/backtesting_summary.csv and write
    report/tables/backtest_table.tex.
    """
    backtest_csv = backtest_csv or os.path.join(
        config.RESULTS_TABLES, "backtesting_summary.csv"
    )
    _check_file(backtest_csv)

    df = pd.read_csv(backtest_csv, index_col=0)

    latex = _df_to_latex(
        df,
        caption=(
            "Backtesting summary: observed vs expected VaR exceedances "
            "and Kupiec test results."
        ),
        label="tab:backtest",
    )
    _save_tex(latex, "backtest_table.tex")
    return latex


def generate_model_comparison_table(model_csv: str = None) -> str:
    """
    Load results/tables/model_comparison.csv and write
    report/tables/model_comparison_table.tex.
    """
    model_csv = model_csv or os.path.join(
        config.RESULTS_TABLES, "model_comparison.csv"
    )
    _check_file(model_csv)

    df = pd.read_csv(model_csv, index_col=0)

    latex = _df_to_latex(
        df,
        caption="Model comparison: average VaR estimates across methods.",
        label="tab:model_comparison",
    )
    _save_tex(latex, "model_comparison_table.tex")
    return latex


# ---------------------------------------------------------------------------
# Figure copy
# ---------------------------------------------------------------------------

def copy_figures_to_report() -> None:
    """
    Copy all .png / .pdf figures from results/figures/ to report/figures/.
    """
    src_dir = config.RESULTS_FIGURES
    dst_dir = config.REPORT_FIGURES
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"[report_generator] No figures directory found at {src_dir}. Skipping.")
        return

    copied = 0
    for fname in os.listdir(src_dir):
        if fname.lower().endswith((".png", ".pdf", ".jpg")):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            print(f"[report_generator] Copied figure: {fname}")
            copied += 1

    print(f"[report_generator] {copied} figure(s) copied to {dst_dir}")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all() -> None:
    """
    Generate all LaTeX table files and copy all figures to report/.
    Call this after running the full analysis pipeline.
    """
    print("=" * 60)
    print("[report_generator] Generating report assets …")
    print("=" * 60)

    _try(generate_latency_table,          "latency table")
    _try(generate_backtest_table,         "backtesting table")
    _try(generate_model_comparison_table, "model comparison table")
    copy_figures_to_report()

    print("\n[report_generator] Done.  Compile the report with:")
    print("  cd report && pdflatex final_report.tex && "
          "bibtex final_report && pdflatex final_report.tex && "
          "pdflatex final_report.tex")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_tex(latex: str, filename: str) -> None:
    os.makedirs(config.REPORT_TABLES, exist_ok=True)
    path = os.path.join(config.REPORT_TABLES, filename)
    with open(path, "w") as f:
        f.write(latex)
    print(f"[report_generator] Saved LaTeX table → {path}")


def _check_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[report_generator] Required file not found: {path}\n"
            "Run the full analysis pipeline first."
        )


def _try(func, label: str) -> None:
    try:
        func()
    except FileNotFoundError as e:
        print(f"[report_generator] Skipping {label}: {e}")


if __name__ == "__main__":
    run_all()
