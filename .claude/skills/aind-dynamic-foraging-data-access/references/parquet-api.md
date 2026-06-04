# Foraging parquet cache — schema & query reference

The cache is the **primary way to access** dynamic-foraging behavior. This file is a condensed
schema + query reference; the **authoritative, self-contained doc is
`src/aind_dynamic_foraging_data_utils/foraging_cache/README.md`** (paste it into an LLM as
context to generate queries). It reads natively over S3 — the bucket is **public, no AWS
credentials needed**.

## S3 layout

```
s3://aind-scratch-data/aind-dynamic-foraging-cache/
├── session_table.parquet                       # flat, ~24k rows × 160 cols
├── trial_table/subject_id=<id>/<id>.parquet     # Hive-partitioned, 1 file/subject (~12.5M × 103)
└── event_table/subject_id=<id>/<id>.parquet     # Hive-partitioned, 1 file/subject (~13.4M × 10)
```

Importable: `from aind_dynamic_foraging_data_utils.foraging_cache import SESSION_DB, TRIAL_DB, EVENT_DB`

## Keys & joins

- Session key: **`_session_id`** in the session table; **`session_id`** in trial/event tables
  (both = `"{subject_id}_{session_date}_{nwb_suffix}"`).
- Trial/event tables are partitioned by `subject_id`; that partition column is inferred as
  **BIGINT**, so use `CAST(subject_id AS VARCHAR)` when filtering / joining / grouping it.

## session_table.parquet (one row per session; 160 cols)

Unions Han-pipeline metadata with the Code Ocean universe. The **~381 CO-only sessions** (absent
from Han) have only identity + CO columns populated; **all Han columns are NULL**.

- **identity**: `_session_id`, `subject_id` (str), `session_date` (`YYYY-MM-DD`), `nwb_suffix`
- **source / quality**: `institute` (`AIND`/`Janelia`), `hardware` (`bonsai`/`bpod`), `rig_type`
  (`training`/`ephys`), `room`; `data_source` (a composite of those); `nwb_data_source`
  (`co_asset`/`bonsai_s3`/`bpod_s3` — the build route, **≠ `data_source`**); `co_asset_id`, `co_s3_nwb_uri`
- **task / curriculum**: `task`, `curriculum_name`, `curriculum_version`, `current_stage_actual`
  (`'None'` = off-curriculum; SQL `NULL` = not in Han)
- **metrics**: `total_trials` (**excludes autowater**), `total_trials_with_autowater`,
  `finished_trials`, `ignored_trials`, `finished_rate`, `reward_trials`, `reward_rate`,
  `foraging_eff`, `foraging_performance`, `bias_naive`, `autowater_*`, `logistic_*` model fits

## trial_table (one row per trial; 103-col union across the three readers)

`session_id`, `subject_id`, `session_date`, `nwb_suffix`, `trial`, `animal_response`
(**0=left / 1=right / 2=ignore**), `earned_reward` (= `rewarded_historyL | rewarded_historyR`,
non-autowater), `rewarded_historyL/R`, `reward_probabilityL/R`, `auto_waterL/R`
(**non-autowater trial = both 0**), `goCue_start_time_in_session`, `choice_time_in_session`,
`reward_time_in_session`, `reaction_time`, `laser_*` (opto, NULL otherwise), `nwb_data_source`.

## event_table (one row per event; 10 cols)

`session_id`, `subject_id`, `session_date`, `nwb_suffix`, `trial`, `timestamps`, `raw_timestamps`,
`event` (one of `goCue_start_time`, `left_lick_time`, `right_lick_time`,
`left_reward_delivery_time`, `right_reward_delivery_time`, `optogenetics_time`), `data`,
`nwb_data_source`.

## Query API — DuckDB (there is no wrapper module)

Always use the three options on the partitioned tables; filter sessions first, then JOIN to
trials via the keys:

```python
import duckdb
from aind_dynamic_foraging_data_utils.foraging_cache import SESSION_DB, TRIAL_DB

READ_TRIALS = f"read_parquet('{TRIAL_DB}/**/*.parquet', hive_partitioning=true, union_by_name=true)"

df = duckdb.sql(f"""
    WITH sel AS (
        SELECT _session_id, subject_id, session_date
        FROM read_parquet('{SESSION_DB}')
        WHERE institute = 'AIND' AND task LIKE '%Uncoupled%'
          AND curriculum_name NOT IN ('None') AND foraging_eff > 0.8
    )
    SELECT s.subject_id, s.session_date, t.session_id,
           t.animal_response, t.earned_reward, t.reward_probabilityL, t.reward_probabilityR
    FROM {READ_TRIALS} t
    JOIN sel s ON t.session_id = s._session_id
    WHERE CAST(t.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
    ORDER BY s.subject_id, s.session_date
""").df()
```

Conventions: **project only the columns you need** (the full trial table is ~21 GB; a few
columns is ~seconds); `total_trials` excludes autowater; `subject_id`/`session_date` are strings;
grouping on `subject_id` also needs the `CAST`. See `foraging_cache/README.md` for the full
schema catalog, common filter columns, worked examples, and an LLM-ready preamble.

## Reading a single NWB directly (bypassing the cache)

`nwb_utils.create_df_trials(nwb_path)` / `create_df_events(nwb_path)` parse one AIND-format NWB
(local path or `s3://`); for Han bonsai/bpod NWBs use the legacy reader in
`foraging_cache/util/nwb_reader_legacy.py`. The cache is built from these readers — for analysis,
query the cache instead (it's ~10,000× faster than opening NWBs one session at a time).
