---
name: aind-foraging-pipeline
description: >
  Knowledge base for the AIND dynamic foraging behavioral data pipeline
  (aind-dynamic-foraging-data-utils repo). Use when working on: NWB file
  processing for dynamic foraging, the parquet database builder/query API,
  session/trial/event table schemas, combining AIND CO assets with Han
  pipeline bonsai/bpod S3 data, or using get_session_table() /
  create_df_trials() / create_df_events() functions. Also use for questions
  about data sources, session metadata joins, schema normalization, or
  the foraging_cache parquet database on S3.
---

# AIND Dynamic Foraging Data Pipeline

## Repo Structure

```
src/aind_dynamic_foraging_data_utils/
├── nwb_utils.py          # AIND NWB readers: load_nwb_from_filename, create_df_trials,
│                         #   create_df_events, create_df_fip, create_df_session
├── code_ocean_utils.py   # CO/docDB API: get_assets, attach_data, add_s3_location,
│                         #   get_foraging_model_info (calls han_pipeline.get_mle_model_fitting)
├── enrich_dfs.py         # FIP enrichment: enrich_fip_in_df_trials, zscore_fip
├── alignment.py          # Event-triggered analysis: event_triggered_response
└── foraging_cache/       # parquet cache builder sub-package
    ├── parquet_builder.py    # core builder: NWB→parquet, reader routing, CO enrich, coalesce
    ├── build_cache.py        # MAIN build entry point: `python -m ...build_cache` (build/update DB)
    ├── query_examples.py     # read-back "return loop": DuckDB query patterns (local or s3://)
    ├── query_examples.ipynb  # DuckDB query notebook
    ├── nwb_reader_aind.py     # AIND reader wrapper (raises AINDReaderQualityError)
    └── nwb_reader_legacy.py   # legacy bonsai/bpod reader (+ h5py fallback for old bpod)
```
Note: there is no `cache_utils.py` — query the parquet cache directly with DuckDB
(`read_parquet(..., hive_partitioning=true, union_by_name=true)`); see `query_examples.py`.

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

## Parquet Database (Issue #135)

See `references/parquet-api.md` for full schema and query API design.

**S3 location**: `s3://aind-scratch-data/aind-dynamic-foraging-cache/`
- `session_table.parquet` — one row per session, merged docDB + han_pipeline metadata
- `trial_table/subject_id=<id>/<subject_id>.parquet` — Hive-partitioned; coalesced one file per subject (default)
- `event_table/subject_id=<id>/<subject_id>.parquet` — Hive-partitioned; coalesced one file per subject
- `build_metadata.json` — tracks processed session IDs for incremental updates
- `processing_log.csv` — human-readable per-session triage log

**Canonical trial schema** (AIND minimal): `session_id`, `subject_id`, `session_date`, `trial`, `animal_response`, `goCue_start_time_in_session`, `choice_time_in_session`, `reward_time_in_session`, `rewarded_historyL`, `rewarded_historyR`, `earned_reward`, `data_source`

## Han Pipeline Architecture (background)

- **Bonsai-basic pipeline** (`aind-foraging-behavior-bonsai-basic`): CO capsule that reads raw NWBs from bonsai S3, runs `compute_df_trial()` + `compute_df_session_meta/performance()`, saves per-session pkl to `foraging_nwb_bonsai_processed/`
- **`df_sessions.pkl`**: Aggregated session table in `foraging_nwb_bonsai_processed/` root, read by `get_session_table()`
- **Note**: Per-session pkl files in `bonsai_processed/` are **incomplete** (still produced, but the pipeline is slated for shutdown and never captured some sessions) → must read raw NWBs for trial data
- **Column mapping** (Han `compute_df_trial()` → AIND canonical): `reward_non_autowater` → `earned_reward`; `goCue_start_time` → subtract t0 for `_in_session` variant
