---
name: aind-dynamic-foraging-data-access
description: >
  How to ACCESS AIND dynamic-foraging behavioral data (the
  aind-dynamic-foraging-data-utils repo): the session / trial / event parquet
  cache on public S3 and how to query it with DuckDB, the table schemas and
  key/filter columns, the data sources (AIND Code Ocean assets + Han bonsai/bpod
  NWBs), and the NWB-reading functions (create_df_trials / create_df_events,
  han_pipeline.get_session_table). Use for questions about querying foraging
  behavior, session / trial / event schemas, session metadata, or where the
  data comes from.
---

# AIND Dynamic Foraging Data Access

AIND dynamic-foraging behavior originates as per-session **NWB files**; for analysis, query the
**parquet cache** built from them (see [Parquet Database](#parquet-database-the-primary-way-to-access-behavior)
below) rather than opening NWBs. The repo is `aind-dynamic-foraging-data-utils`; the cache lives in
its `foraging_cache/` sub-package. Query it with the helpers in `foraging_cache.query`
(`select_sessions` → `fetch_trials`/`fetch_events`; DuckDB under the hood), and drop to native
SQL via `read_trials`/`read_events` when you need more. **Authoritative docs:**
`foraging_cache/README.md` (querying — self-contained, paste into an LLM as context) and
`README_build.md` (building).

## Data Sources

Five data sources, each with different coverage and format. See `references/data-sources.md` for full details.

| Priority | Source | Coverage | Format | Key Path |
|----------|--------|----------|--------|----------|
| 1 (best) | AIND CO assets | ~2/3 sessions | New NWB (zarr) | `/data/{asset_name}/nwb/{session}.nwb` or S3 |
| 2 | Han bonsai S3 | ~all sessions | Old NWB (HDF5) | `s3://aind-behavior-data/foraging_nwb_bonsai/` |
| 3 | Han bpod S3 | Older sessions | Old NWB (HDF5) | `s3://aind-behavior-data/foraging_nwb_bpod/` |
| Meta | han_pipeline session table | All sessions | DataFrame | `aind_analysis_arch_result_access.han_pipeline.get_session_table(if_load_bpod=True)` |
| Meta | docDB | ~2/3 sessions | MongoDB | `MetadataDbClient` via `get_assets()` in code_ocean_utils.py |

## Key Functions to Reuse

**NWB reading (AIND format, `nwb_utils.py`)**
- `create_df_trials(nwb_path)` → trial-level DataFrame (AIND canonical schema)
- `create_df_events(nwb_path)` → tidy event DataFrame
- `load_nwb_from_filename(path)` → auto-detects HDF5 vs zarr, supports s3://

**docDB querying (`code_ocean_utils.py`)**
- `get_assets(subjects=[], processed=True, modality=["behavior"])` → session metadata + CO asset IDs
- `add_s3_location(results)` → adds s3:// NWB URLs to results DataFrame

**Han pipeline (`aind_analysis_arch_result_access.han_pipeline`)**
- `get_session_table(if_load_bpod=True)` → master session table with foraging_eff, finished_rate, curriculum, etc.
- `get_mle_model_fitting(subject_id, session_date, agent_alias)` → RL model fits

## Session Identity

Canonical key: `(subject_id, session_date, nwb_suffix)` as a 3-tuple.

- `ses_idx` used throughout codebase = `f"{subject_id}_{session_date}"` (without nwb_suffix)
- Multiple sessions per day handled via `nwb_suffix` (0, 1, 2... or HHMMSS integer)
- Bpod sessions: `'bpod' in nwb.session_description` is True

## Parquet Database (the primary way to access behavior)

A Hive-partitioned parquet cache of all dynamic-foraging behavior on a **public** S3 bucket (no
AWS credentials needed to read). **Authoritative, self-contained querying doc:**
`foraging_cache/README.md` — paste it into an LLM as context to generate queries; full schema +
query patterns in `references/parquet-api.md`.

**S3 location**: `s3://aind-scratch-data/aind-dynamic-foraging-cache/` (paths `SESSION_DB` /
`TRIAL_DB` / `EVENT_DB` and the query helpers below are importable from
`aind_dynamic_foraging_data_utils.foraging_cache`).
- `session_table.parquet` — one row per session (~24k × 160 cols). Unions Han metadata with the
  Code Ocean universe; the ~381 CO-only sessions absent from Han have all Han columns NULL.
- `trial_table/subject_id=<id>/<subject_id>.parquet` — Hive-partitioned, 1 file/subject (~12.5M × 103 cols)
- `event_table/subject_id=<id>/<subject_id>.parquet` — Hive-partitioned, 1 file/subject

**Querying — use the helpers first (`foraging_cache.query`):**
```python
from aind_dynamic_foraging_data_utils.foraging_cache import select_sessions, fetch_trials, fetch_events
sel    = select_sessions("task LIKE '%Uncoupled%' AND foraging_eff > 0.8")  # filter the session table
trials = fetch_trials(sel, columns=["animal_response", "earned_reward"])     # their trials + metadata joined
```
- `select_sessions(where=..., subjects=..., columns=...)` → filtered session DataFrame (covers both
  "metric filter → fetch" and "subject → session → fetch"); `fetch_trials`/`fetch_events(sel, ...)`
  read **only the selected subjects' partitions** (~1 s) and join the session metadata on.
- More than the helpers cover (aggregations, windows, trial↔event joins)? `read_trials(subjects)` /
  `read_events(subjects)` return a fast partition-scoped `read_parquet(...)` clause to drop into any SQL.

**Native SQL conventions (what the helpers do under the hood):**
- session key is `_session_id` (session table) ↔ `session_id` (trial/event)
- always `hive_partitioning=true, union_by_name=true` + `CAST(subject_id AS VARCHAR)` (partition col is inferred BIGINT)
- scope the read to the subjects you need (a `subject_id=<id>/*.parquet` glob/list) — a full
  `/**/*.parquet` glob reads every subject's footer to build the union (~25 s cold)
- trial table is a **103-column union** across the three readers; key cols: `animal_response`
  (0=L/1=R/2=ignore), `earned_reward`, `reward_probabilityL/R`, `rewarded_historyL/R`,
  `auto_waterL/R`, `goCue_start_time_in_session`
- session filters: `institute`/`hardware`/`rig_type`, `task`, `curriculum_name` (`'None'` =
  off-curriculum), `current_stage_actual` (curriculum stage `STAGE_1_WARMUP`…`STAGE_FINAL`/`GRADUATED`;
  **"Final stages" = `STAGE_FINAL` OR `GRADUATED`** — same training params, so use
  `current_stage_actual IN ('STAGE_FINAL','GRADUATED')` for fully-trained sessions), metrics
  (`foraging_eff`, `finished_trials`, …). NB `data_source` (Han rig/institute composite) ≠
  `nwb_data_source` (`co_asset`/`bonsai_s3`/`bpod_s3`).
- **`curriculum_name` ≠ `task`**: `curriculum_name` = the training *program* (named after its
  **target task**, constant as the mouse progresses); `task` = the paradigm *actually run that
  session*, which changes by stage as the curriculum ramps difficulty (e.g. Uncoupled Baiting
  curriculum runs Coupled Baiting in early stages → Uncoupled Baiting at STAGE_3→FINAL/GRADUATED).
  Filter `curriculum_name` for enrollment, `task` for what ran (~3.2k on-curriculum sessions differ).

(To build/update the cache, see `build_cache.py` / `README_build.md` — not needed for analysis.)

## Han Pipeline Architecture (background)

- **Bonsai-basic pipeline** (`aind-foraging-behavior-bonsai-basic`): CO capsule that reads raw NWBs from bonsai S3, runs `compute_df_trial()` + `compute_df_session_meta/performance()`, saves per-session pkl to `foraging_nwb_bonsai_processed/`
- **`df_sessions.pkl`**: Aggregated session table in `foraging_nwb_bonsai_processed/` root, read by `get_session_table()`
- **Note**: Per-session pkl files in `bonsai_processed/` are **incomplete** (still produced, but the pipeline is slated for shutdown and never captured some sessions) → must read raw NWBs for trial data
- **Column mapping** (Han `compute_df_trial()` → AIND canonical): `reward_non_autowater` → `earned_reward`; `goCue_start_time` → subtract t0 for `_in_session` variant
