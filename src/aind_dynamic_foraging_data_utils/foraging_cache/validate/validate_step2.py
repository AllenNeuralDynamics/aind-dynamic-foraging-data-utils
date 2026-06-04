"""
Validation step 2: aggregate consistency of the cache vs Han's master session
table. Compares per-session trial count and rewarded-trial count, totals, and
explains where/why they diverge.

Cache  : COUNT(*) trials and SUM(earned_reward) per session_id.
Han    : total_trials / finished_trials / reward_trials per session.
Artifacts -> scratch/tmp/validation/step2_*.csv
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, duckdb

OUT = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
TRIALS = f"read_parquet('{OUT}/trial_table/**/*.parquet', hive_partitioning=true, union_by_name=true)"
ART = "/root/capsule/scratch/tmp/validation"


def main():
    print("== cache per-session aggregates ==", flush=True)
    cache = duckdb.sql(f"""
        SELECT session_id,
               COUNT(*)                                              AS n_trials_cache,
               SUM(CASE WHEN TRY_CAST(earned_reward AS BOOLEAN) THEN 1 ELSE 0 END) AS n_rew_cache
        FROM {TRIALS} GROUP BY session_id
    """).df()
    print(f"  cache: {len(cache)} sessions | total trials = {cache.n_trials_cache.sum():,} "
          f"| total rewarded (earned) = {cache.n_rew_cache.sum():,}", flush=True)

    sess = pd.read_parquet(f"{OUT}/session_table.parquet")
    han = sess[["_session_id", "nwb_data_source", "total_trials", "finished_trials",
                "reward_trials"]].copy()

    m = cache.merge(han, left_on="session_id", right_on="_session_id", how="outer", indicator=True)
    in_both = m[m._merge == "both"].copy()
    cache_only = m[m._merge == "left_only"]
    han_only = m[m._merge == "right_only"]
    print(f"\n  sessions: both={len(in_both)}  cache-only={len(cache_only)}  "
          f"han-only(skipped/failed)={len(han_only)}", flush=True)

    # ---- totals over matched sessions ----
    print("\n== totals over MATCHED sessions ==", flush=True)
    print(f"  trials   : cache={in_both.n_trials_cache.sum():,.0f}  "
          f"Han.total_trials={in_both.total_trials.sum():,.0f}  "
          f"Han.finished_trials={in_both.finished_trials.sum():,.0f}", flush=True)
    print(f"  rewarded : cache.earned={in_both.n_rew_cache.sum():,.0f}  "
          f"Han.reward_trials={in_both.reward_trials.sum():,.0f}", flush=True)

    # ---- per-session agreement ----
    in_both["d_total"] = in_both.n_trials_cache - in_both.total_trials
    in_both["d_finished"] = in_both.n_trials_cache - in_both.finished_trials
    in_both["d_rew"] = in_both.n_rew_cache - in_both.reward_trials
    print("\n== per-session: cache n_trials vs Han ==", flush=True)
    print(f"  == Han.total_trials   : {(in_both.d_total == 0).sum()}/{len(in_both)}", flush=True)
    print(f"  == Han.finished_trials: {(in_both.d_finished == 0).sum()}/{len(in_both)}", flush=True)
    print(f"  == Han.reward_trials  : {(in_both.d_rew == 0).sum()}/{len(in_both)}", flush=True)
    print("\n  cache n_trials - Han.total_trials  (describe):\n" +
          in_both.d_total.describe().round(2).to_string(), flush=True)
    print("\n  match rate vs total_trials by source:", flush=True)
    print(in_both.groupby("nwb_data_source").apply(
        lambda g: pd.Series({"sessions": len(g),
                             "==total": int((g.d_total == 0).sum()),
                             "==finished": int((g.d_finished == 0).sum()),
                             "==reward": int((g.d_rew == 0).sum())})).to_string(), flush=True)
    in_both.to_csv(f"{ART}/step2_per_session.csv", index=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
