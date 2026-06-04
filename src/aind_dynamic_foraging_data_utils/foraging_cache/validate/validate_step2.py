"""
Validation step 2: APPLES-TO-APPLES consistency of the parquet cache vs Han's
master session table, computed straight from the cache trial table.

Han's master table is derived ONLY from the bonsai/bpod NWBs, and its session
stats are specific *sums over df_trial* (process_nwbs.py / compute_df_*). The
crucial subtlety: Han's ``total_trials`` EXCLUDES autowater trials, while a naive
cache ``COUNT(*)`` includes them. So we reproduce each Han stat from the cache
with the SAME definition and compare like-for-like:

  non_aw   = (auto_waterL == 0) AND (auto_waterR == 0)        # process_nwbs L80
  IGNORE   = 2  (animal_response: LEFT=0, RIGHT=1, IGNORE=2)  # L302

  metric                          cache (trial table)                 Han col
  total_trials (incl. autowater)  COUNT(*)                            total_trials_with_autowater
  total_trials (foraging, non-AW) SUM(non_aw)                         total_trials
  finished_trials (non-AW)        SUM(non_aw & resp!=IGNORE)          finished_trials
  autowater_trials                SUM(NOT non_aw)                     autowater_collected+ignored
  reward_trials (earned, non-AW)  SUM(rewarded_historyL|R)            reward_trials
  left_choices  (non-AW)          SUM(non_aw & resp==LEFT)            <- bias_naive,finished
  right_choices (non-AW)          SUM(non_aw & resp==RIGHT)           <- bias_naive,finished

Han doesn't store n_left/n_right directly; recover them exactly from
``bias_naive = 2*(n_right/(n_left+n_right) - 0.5)`` with n_left+n_right =
finished_trials (process_nwbs L323-325):
  n_right = round((bias_naive+1)/2 * finished_trials);  n_left = finished - n_right

Artifacts -> scratch/tmp/validation/step2_apples.csv
"""
import warnings; warnings.simplefilter("ignore")
import pandas as pd, duckdb

OUT = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
TR = f"read_parquet('{OUT}/trial_table/**/*.parquet', hive_partitioning=true, union_by_name=true)"
ART = "/root/capsule/scratch/tmp/validation"

# (label, cache column, Han column)
PAIRS = [
    ("total_trials (incl. autowater)", "c_all",         "total_trials_with_autowater"),
    ("total_trials (foraging, non-AW)", "c_total_nonaw", "total_trials"),
    ("finished_trials (non-AW)",        "c_finished",    "finished_trials"),
    ("autowater_trials",                "c_autowater",   "han_autowater"),
    ("reward_trials (earned, non-AW)",  "c_reward",      "reward_trials"),
    ("left_choices (non-AW)",           "c_left",        "han_left"),
    ("right_choices (non-AW)",          "c_right",       "han_right"),
]


def main():
    print("== cache per-session stats (reproducing Han's definitions) ==", flush=True)
    cache = duckdb.sql(f"""
        WITH t AS (
            SELECT session_id,
                   COALESCE(TRY_CAST(auto_waterL AS DOUBLE), 0) AS awL,
                   COALESCE(TRY_CAST(auto_waterR AS DOUBLE), 0) AS awR,
                   TRY_CAST(animal_response AS INTEGER)         AS ar,
                   COALESCE(TRY_CAST(rewarded_historyL AS BOOLEAN), FALSE) AS rL,
                   COALESCE(TRY_CAST(rewarded_historyR AS BOOLEAN), FALSE) AS rR
            FROM {TR}
        )
        SELECT session_id,
               COUNT(*)                                                  AS c_all,
               SUM(CASE WHEN awL=0 AND awR=0           THEN 1 ELSE 0 END) AS c_total_nonaw,
               SUM(CASE WHEN awL=0 AND awR=0 AND ar<>2 THEN 1 ELSE 0 END) AS c_finished,
               SUM(CASE WHEN NOT (awL=0 AND awR=0)     THEN 1 ELSE 0 END) AS c_autowater,
               SUM(CASE WHEN rL OR rR                  THEN 1 ELSE 0 END) AS c_reward,
               SUM(CASE WHEN awL=0 AND awR=0 AND ar=0  THEN 1 ELSE 0 END) AS c_left,
               SUM(CASE WHEN awL=0 AND awR=0 AND ar=1  THEN 1 ELSE 0 END) AS c_right
        FROM t GROUP BY session_id
    """).df()
    print(f"  {len(cache)} sessions aggregated from the cache trial table", flush=True)

    sess = pd.read_parquet(f"{OUT}/session_table.parquet")
    hcols = ["_session_id", "nwb_data_source", "total_trials_with_autowater", "total_trials",
             "finished_trials", "autowater_collected", "autowater_ignored", "reward_trials",
             "bias_naive"]
    m = cache.merge(sess[hcols], left_on="session_id", right_on="_session_id", how="inner")
    m = m.dropna(subset=["total_trials"]).copy()   # keep sessions Han actually has stats for

    m["han_autowater"] = m["autowater_collected"].fillna(0) + m["autowater_ignored"].fillna(0)
    fin = m["finished_trials"]; bias = m["bias_naive"]
    m["han_right"] = ((bias + 1) / 2 * fin).round()
    m["han_left"] = fin - m["han_right"]

    print(f"\n== APPLES-TO-APPLES (n={len(m)} sessions in both cache & Han) ==", flush=True)
    hdr = (f"{'metric':32s} {'exact%':>7} {'|d|<=1%':>8} {'medΔ':>5} {'meanΔ':>7}  "
           f"{'cache_total':>13} {'Han_total':>13}")
    print(hdr); print("-" * len(hdr), flush=True)
    out = m[["session_id", "nwb_data_source"]].copy()
    for label, cc, hc in PAIRS:
        d = (m[cc] - m[hc])
        v = d.dropna()
        out[f"d_{cc}"] = d
        print(f"{label:32s} {100*(v==0).mean():7.1f} {100*(v.abs()<=1).mean():8.1f} "
              f"{v.median():5.0f} {v.mean():7.2f}  {m[cc].sum():13,.0f} {m[hc].sum():13,.0f}",
              flush=True)

    # broken down by route (Han is bonsai/bpod-derived; co_asset is cross-pipeline)
    print("\n== exact-match % by source, headline 'total_trials (incl. autowater)' ==", flush=True)
    m["d_all"] = m["c_all"] - m["total_trials_with_autowater"]
    print(m.groupby("nwb_data_source")
          .apply(lambda g: pd.Series({"n": len(g), "exact%": round(100 * (g.d_all == 0).mean(), 1),
                                       "|d|<=1%": round(100 * (g.d_all.abs() <= 1).mean(), 1)}))
          .to_string(), flush=True)

    out.to_csv(f"{ART}/step2_apples.csv", index=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
