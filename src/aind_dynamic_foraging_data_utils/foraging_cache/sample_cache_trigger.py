"""
Build parquet tables from N randomly sampled sessions across all dates.

Strategy: build the Han session table *without* CO assets first (fast),
filter to sessions that have a local NWB, sample, then enrich only the
sampled rows with CO asset metadata from docDB.

Sessions are drawn from all available sources:
  - CO assets  (docDB S3 URIs, highest priority)
  - Bonsai S3  (local /data/foraging_nwb_bonsai/)
  - bpod S3    (local /data/foraging_nwb_bpod/)

Outputs (local):
    /root/capsule/scratch/foraging_cache/session_table_sample.parquet
    /root/capsule/scratch/foraging_cache/trial_table/
    /root/capsule/scratch/foraging_cache/event_table/
    /root/capsule/scratch/foraging_cache/build_metadata.json
"""

import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/src")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from aind_dynamic_foraging_data_utils.foraging_cache import parquet_builder

# ---- Output paths ----
SCRATCH      = "/root/capsule/scratch/foraging_cache"
SESSION_OUT  = f"{SCRATCH}/session_table_sample.parquet"
TRIAL_OUT    = f"{SCRATCH}/trial_table"
EVENT_OUT    = f"{SCRATCH}/event_table"
META_OUT     = f"{SCRATCH}/build_metadata.json"

N_SAMPLE     = 100
RANDOM_SEED  = 42

os.makedirs(SCRATCH, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Build NWB file index from local directories
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Building NWB file index from local directories")
print("=" * 60)
nwb_index = parquet_builder.build_nwb_file_index(
    bonsai_dir=parquet_builder.LOCAL_BONSAI_NWB_DIR,
    bpod_dir=parquet_builder.LOCAL_BPOD_NWB_DIR,
)
print(f"  Total NWB files indexed: {len(nwb_index)}")

# ---------------------------------------------------------------------------
# 2.  Build session table WITHOUT CO assets (fast — Han table only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Building session table (Han only, no CO assets)")
print("=" * 60)

df_all = parquet_builder.build_session_table(
    output_path=SESSION_OUT.replace("_sample", "_full"),
    bowen_csv_path=parquet_builder.BOWEN_INCOMPLETE_CSV,
    include_co_assets=False,
    verbose=True,
    incremental=False,
)

print(f"  Total sessions: {len(df_all)}")
print(f"  Date range    : {df_all['session_date'].min()} -> {df_all['session_date'].max()}")

# ---------------------------------------------------------------------------
# 3.  Filter to sessions that have a local NWB file
#     (CO assets will be looked up later, only for the sample)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Filtering to sessions with local NWB files")
print("=" * 60)

def _has_local_nwb(row):
    suffix = int(row["nwb_suffix"]) if pd.notna(row["nwb_suffix"]) else -1
    return (str(row["subject_id"]), str(row["session_date"]), suffix) in nwb_index

has_local = df_all.apply(_has_local_nwb, axis=1)
not_bad   = ~df_all["is_bad_bowen_session"]

df_avail = df_all[not_bad & has_local].copy()

print(f"  Bad Bowen sessions excluded : {(~not_bad).sum()}")
print(f"  Sessions with local NWB     : {has_local.sum()}")
print(f"  Total processable           : {len(df_avail)}")

# ---------------------------------------------------------------------------
# 4.  Stratified random sample of N_SAMPLE sessions across all dates
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Step 4: Stratified random sample of {N_SAMPLE} sessions across all dates")
print("=" * 60)

rng = np.random.default_rng(RANDOM_SEED)

date_counts = df_avail["session_date"].value_counts()
n_dates = len(date_counts)
print(f"  Unique dates in pool: {n_dates}")

# Proportional allocation per date
total = len(df_avail)
quotas = (date_counts / total * N_SAMPLE).round().astype(int)
diff = N_SAMPLE - quotas.sum()
if diff > 0:
    remainders = (date_counts / total * N_SAMPLE) - quotas
    quotas[remainders.nlargest(abs(diff)).index] += 1
elif diff < 0:
    quotas[quotas.nlargest(abs(diff)).index] -= 1

sampled_frames = []
for date, quota in quotas.items():
    if quota <= 0:
        continue
    pool = df_avail[df_avail["session_date"] == date]
    n_pick = min(quota, len(pool))
    sampled_frames.append(pool.sample(n=n_pick, random_state=int(rng.integers(1e9))))

df_sample = pd.concat(sampled_frames, ignore_index=True)
print(f"  Sample size : {len(df_sample)} sessions across {df_sample['session_date'].nunique()} dates")
print(f"  Date range  : {df_sample['session_date'].min()} -> {df_sample['session_date'].max()}")

# ---------------------------------------------------------------------------
# 5.  Enrich only the sampled rows with CO asset metadata from docDB
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Step 5: Enriching {len(df_sample)} sampled sessions with CO assets")
print("=" * 60)

sample_subjects = sorted(df_sample["subject_id"].unique().tolist())
print(f"  Unique subjects in sample: {len(sample_subjects)}")

df_sample = parquet_builder._enrich_with_co_assets(
    df_sample, sample_subjects, verbose=True,
)

# Refresh nwb_data_source now that CO assets are populated
df_sample["nwb_data_source"] = df_sample.apply(
    parquet_builder._assign_nwb_data_source, axis=1
)

has_co = df_sample["co_s3_nwb_uri"].notna() & (df_sample["co_s3_nwb_uri"] != "")
print(f"  With CO asset : {has_co.sum()}")
print(f"  Local only    : {(~has_co).sum()}")

# ---------------------------------------------------------------------------
# 6.  Write sample session table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Step 6: Writing sample session table -> {SESSION_OUT}")
print("=" * 60)
parquet_builder._write_dataframe_as_parquet(df_sample, SESSION_OUT)
print(f"  Written: {len(df_sample)} rows")

# ---------------------------------------------------------------------------
# 7.  Build trial + event tables
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Building trial and event parquet tables")
print("=" * 60)

summary = parquet_builder.build_trial_and_event_tables(
    session_df=df_sample,
    trial_output_prefix=TRIAL_OUT,
    event_output_prefix=EVENT_OUT,
    nwb_file_index=nwb_index,
    build_metadata_path=META_OUT,
    incremental=False,
    max_sessions=None,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 8.  Final summary
# ---------------------------------------------------------------------------
trial_files = glob.glob(f"{TRIAL_OUT}/**/*.parquet", recursive=True)
event_files = glob.glob(f"{EVENT_OUT}/**/*.parquet", recursive=True)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\n  [Session pool]")
print(f"    Han table total              : {len(df_all)}")
print(f"    Bad Bowen sessions excluded  : {(~not_bad).sum()}")
print(f"    Processable (local NWB)      : {len(df_avail)}")
print(f"    Date range                   : {df_avail['session_date'].min()} -> {df_avail['session_date'].max()}")
print(f"    Unique dates in pool         : {df_avail['session_date'].nunique()}")

print(f"\n  [Sample  (n={len(df_sample)})]")
print(f"    Unique dates covered         : {df_sample['session_date'].nunique()}")
print(f"    Date range                   : {df_sample['session_date'].min()} -> {df_sample['session_date'].max()}")
print(f"    With CO asset                : {has_co.sum()}")
print(f"    Local only                   : {(~has_co).sum()}")

print(f"\n  [Build results]")
print(f"    Processed (ok)               : {summary['n_processed']}")
print(f"    Skipped (no NWB found)       : {summary['n_skipped']}")
print(f"    Failed                       : {summary['n_failed']}")
print(f"\n    Data source breakdown:")
print(f"      CO asset                   : {summary['n_co_asset']}")
print(f"      Bonsai S3                  : {summary['n_bonsai_s3']}")
print(f"      bpod S3                    : {summary['n_bpod_s3']}")
print(f"\n    NWB reader breakdown:")
print(f"      AIND data-util (direct)    : {summary['n_aind_reader']}")
print(f"      AIND->legacy fallback      : {summary['n_aind_fallback_legacy']}")
print(f"      Legacy bpod (direct)       : {summary['n_legacy_bpod']}")

if summary["failed_sessions"]:
    print(f"\n    Failed sessions ({summary['n_failed']}):")
    for fs in summary["failed_sessions"][:20]:
        print(f"      [{fs.get('data_source','?')}] {fs['session_id']}  --  {fs['error']}")
    if summary["n_failed"] > 20:
        print(f"      ... and {summary['n_failed'] - 20} more")

print(f"\n  [Output files]")
print(f"    Session table parquet        : {SESSION_OUT}")
print(f"    Trial parquet files          : {len(trial_files)}  ->  {TRIAL_OUT}")
print(f"    Event parquet files          : {len(event_files)}  ->  {EVENT_OUT}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 9.  Example queries (DuckDB)
# ---------------------------------------------------------------------------
import duckdb

print("\n" + "=" * 60)
print("EXAMPLE QUERIES")
print("=" * 60)

# ---- Query 1: Session-level query on the session table ----
# Example: high-performing Uncoupled sessions (foraging_eff > 0.8)
print("\n--- Session query: Uncoupled tasks with foraging_eff > 0.8 ---")
selected = duckdb.sql(f"""
    SELECT _session_id, subject_id, session_date, finished_trials, foraging_eff, task
    FROM read_parquet('{SESSION_OUT}')
    WHERE task LIKE '%Uncoupled%'
      AND foraging_eff > 0.8
    ORDER BY session_date, subject_id
""").df()
print(selected.to_string(index=False))

# ---- Query 2: Trial history — session filter drives JOIN, session keys merged in ----
print(f"\n--- Trial history for {len(selected)} selected sessions ---")
df_trials_all = duckdb.sql(f"""
    WITH sel AS (
        SELECT _session_id, subject_id, session_date, task, foraging_eff
        FROM read_parquet('{SESSION_OUT}')
        WHERE task LIKE '%Uncoupled%' AND foraging_eff > 0.8
    )
    SELECT s.subject_id, s.session_date, s.task, s.foraging_eff,
           t.session_id, t.animal_response, t.earned_reward,
           t.reward_probabilityL, t.reward_probabilityR,
           t.rewarded_historyL, t.rewarded_historyR
    FROM read_parquet('{TRIAL_OUT}/**/*.parquet', hive_partitioning=true, union_by_name=true) t
    JOIN sel s ON t.session_id = s._session_id
    WHERE CAST(t.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
    ORDER BY s.subject_id, s.session_date
""").df()
print(f"  Total trials across {len(selected)} sessions : {len(df_trials_all)}")
print(df_trials_all.head(10).to_string(index=False))

# ---- Query 3: Event history — same pattern ----
print(f"\n--- Event history for {len(selected)} selected sessions ---")
df_events_all = duckdb.sql(f"""
    WITH sel AS (
        SELECT _session_id, subject_id, session_date, task, foraging_eff
        FROM read_parquet('{SESSION_OUT}')
        WHERE task LIKE '%Uncoupled%' AND foraging_eff > 0.8
    )
    SELECT s.subject_id, s.session_date, e.session_id, e.timestamps, e.event, e.data
    FROM read_parquet('{EVENT_OUT}/**/*.parquet', hive_partitioning=true, union_by_name=true) e
    JOIN sel s ON e.session_id = s._session_id
    WHERE CAST(e.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
    ORDER BY s.subject_id, s.session_date, e.timestamps
""").df()
print(f"  Total events across {len(selected)} sessions : {len(df_events_all)}")
print(f"  Event types : {sorted(df_events_all['event'].unique().tolist())}")
print(df_events_all.head(10).to_string(index=False))
