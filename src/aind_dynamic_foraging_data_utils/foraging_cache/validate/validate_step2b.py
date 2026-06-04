"""
Validation step 2b: apples-to-apples trial-count delta (cache TOTAL trials vs Han
TOTAL trials), its distribution, and its evolution over time.

Context: Han's master table is derived EXCLUSIVELY from the bonsai/bpod NWBs. So:
  - cache bonsai_s3 / bpod_s3 sessions  -> SAME NWB family as Han  => true
    apples-to-apples; delta should be ~0 (only reader-version differences).
  - cache co_asset sessions             -> AIND CO NWB, while Han counts the
    bonsai NWB of the same session  => CROSS-PIPELINE; delta is expected, not a
    cache error.

delta = cache_total_trials - han_total_trials (per session).
Artifacts -> scratch/tmp/validation/step2_delta_*.{csv,png}
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
ART = "/root/capsule/scratch/tmp/validation"
COLORS = {"co_asset": "#1f77b4", "bonsai_s3": "#ff7f0e", "bpod_s3": "#2ca02c"}


def dist_row(g):
    d = g.d_total
    return pd.Series({
        "n": len(d),
        "median": d.median(), "mean": round(d.mean(), 2), "std": round(d.std(), 1),
        "==0 %": round(100 * (d == 0).mean(), 1),
        "|d|<=1 %": round(100 * (d.abs() <= 1).mean(), 1),
        "|d|<=5 %": round(100 * (d.abs() <= 5).mean(), 1),
        "|d|>50 %": round(100 * (d.abs() > 50).mean(), 1),
        "min": d.min(), "max": d.max(),
    })


def main():
    m = pd.read_csv(f"{ART}/step2_per_session.csv")
    m = m[m._merge == "both"].copy()
    # attach session_date from the session table
    sess = pd.read_parquet(f"{OUT}/session_table.parquet")[["_session_id", "session_date"]]
    m = m.merge(sess, on="_session_id", how="left")
    m["session_date"] = pd.to_datetime(m["session_date"], errors="coerce")
    m = m.dropna(subset=["total_trials", "session_date", "d_total"])
    m["d_total"] = m["d_total"].astype(float)
    m["rel"] = m["d_total"] / m["total_trials"]

    print("== delta = cache_total - Han_total (apples-to-apples: total vs total) ==", flush=True)
    print("\nBY SOURCE:\n" + m.groupby("nwb_data_source").apply(dist_row).to_string(), flush=True)
    same = m[m.nwb_data_source.isin(["bonsai_s3", "bpod_s3"])]
    cross = m[m.nwb_data_source == "co_asset"]
    print(f"\nSAME-NWB (bonsai+bpod, true apples-to-apples): n={len(same)} "
          f"median={same.d_total.median():.0f} ==0:{100*(same.d_total==0).mean():.1f}% "
          f"|d|<=1:{100*(same.d_total.abs()<=1).mean():.1f}%", flush=True)
    print(f"CROSS-PIPELINE (co_asset vs Han bonsai):        n={len(cross)} "
          f"median={cross.d_total.median():.0f} ==0:{100*(cross.d_total==0).mean():.1f}% "
          f"|d|<=1:{100*(cross.d_total.abs()<=1).mean():.1f}%", flush=True)
    print("\nDELTA DECILES (overall):\n" +
          m.d_total.quantile([0, .1, .25, .5, .75, .9, .99, 1]).round(1).to_string(), flush=True)
    m.to_csv(f"{ART}/step2_delta_detail.csv", index=False)

    # ---- figure: delta vs time (historical view) + distribution ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.7, 1]})
    for src, g in m.groupby("nwb_data_source"):
        axA.scatter(g.session_date, g.d_total, s=5, alpha=0.12, color=COLORS[src], edgecolors="none")
        med = g.set_index("session_date").d_total.resample("MS").median()
        axA.plot(med.index, med.values, color=COLORS[src], lw=2.2, label=f"{src} (monthly median)")
    axA.axhline(0, color="k", lw=0.9)
    axA.set_yscale("symlog", linthresh=10)
    axA.set_ylim(-700, 700)
    axA.set_xlabel("session date"); axA.set_ylabel("delta trials  (cache_total - Han_total)")
    axA.set_title("Trial-count delta over time (symlog y)\npositive = cache has MORE trials than Han")
    axA.legend(fontsize=8, loc="upper left"); axA.grid(True, alpha=.3)

    bins = np.linspace(-60, 60, 61)
    for src, g in m.groupby("nwb_data_source"):
        axB.hist(g.d_total.clip(-60, 60), bins=bins, alpha=0.55, color=COLORS[src],
                 label=f"{src} (n={len(g)})")
    axB.axvline(0, color="k", lw=0.9)
    axB.set_yscale("log")
    axB.set_xlabel("delta trials (clipped to ±60)"); axB.set_ylabel("# sessions (log)")
    axB.set_title("Delta distribution by source")
    axB.legend(fontsize=8); axB.grid(True, alpha=.3)
    plt.tight_layout()
    out = f"{ART}/step2_delta_vs_time.png"; fig.savefig(out, dpi=110)
    print(f"\nsaved {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
