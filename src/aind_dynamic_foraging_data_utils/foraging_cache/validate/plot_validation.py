"""
Plot cache-vs-legacy fetch timing from the CSVs written by validate_step1.py.
Decoupled from the (slow) measurement run so the figure can be regenerated/tuned
cheaply. Uses explicit axis limits + tight_layout (NOT bbox_inches='tight', which
blows up the canvas under a wide log-scale autoscale).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ART = "/root/capsule/scratch/tmp/validation"
N_FULL = 23582
OPS = ["trial_5col", "trial_full", "trial+event_full"]


def main():
    cache = pd.read_csv(f"{ART}/cache_timing.csv")
    legacy = pd.read_csv(f"{ART}/legacy_timing.csv")
    Nline = np.array([1, 10, 100, 1000, 10000, N_FULL], float)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ymax = 1.0
    for op in OPS:
        g = cache[cache.op == op].sort_values("N")
        col = None
        if len(g):
            ln, = ax.plot(g.N, g.sec, "o-", lw=2, label=f"cache · {op}")
            col = ln.get_color(); ymax = max(ymax, g.sec.max())
        per = legacy[(legacy.op == op) & (legacy.source == "co_asset")]["per_session"].mean()
        if np.isfinite(per) and per > 0:
            ax.plot(Nline, per * Nline, "s--", color=col, alpha=.6,
                    label=f"legacy CO chain · {op} (~{per:.0f} s/sess, extrap.)")
            ymax = max(ymax, per * N_FULL)
    ax.axvline(N_FULL, ls=":", c="gray", alpha=.5)
    ax.annotate("full DB", (N_FULL, ymax), fontsize=8, color="gray", ha="right", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, N_FULL * 2)
    ax.set_ylim(0.05, ymax * 3)
    ax.set_xlabel("# sessions fetched"); ax.set_ylabel("time (s)")
    ax.set_title("Fetch time: parquet cache (DuckDB/S3, measured)\nvs TRUE legacy CO route "
                 "(docDB query + S3 glob + open NWB, serial — extrapolated)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=.3)
    plt.tight_layout()
    out = f"{ART}/cache_vs_legacy.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
