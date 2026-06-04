# Parquet Database Schema and Query API

## Overview

GitHub Issue: https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils/issues/135

Goal: Centralized cached parquet database on S3 combining all data sources, with a clean query API in this repo.

## S3 Layout

```
s3://aind-behavior-data/foraging_cache/
├── session_table.parquet              # single file; ~all sessions; fast full load
├── trial_table/
│   └── subject_id=<id>/              # Hive-partitioned; use filters for per-mouse queries
│       └── part.0.parquet
├── event_table/
│   └── subject_id=<id>/              # Hive-partitioned
│       └── part.0.parquet
└── build_metadata.json               # {"processed_sessions": [...], "last_built": "..."}
```

## Table Schemas

### session_table.parquet

One row per session. Primary key: `session_id = f"{subject_id}_{session_date}_{nwb_suffix}"`.

**From `get_session_table(if_load_bpod=True)` (primary)**:
- `subject_id`, `session_date`, `nwb_suffix`, `session` (ordinal session number)
- `rig`, `trainer`, `PI`, `task`, `notes`
- `curriculum_name`, `curriculum_version`, `current_stage_actual`
- `finished_trials`, `finished_rate`, `foraging_eff`, `foraging_eff_random_seed`, `foraging_performance`
- `bias_naive`, `reaction_time_median`, `early_lick_rate`, `double_dipping_rate_finished_trials`
- `weight_after`, `weight_after_ratio`, `water_in_session_total`
- `duration_iti_median`, `duration_iti_mean`, `duration_gocue_stop_median`

**Added by builder**:
- `session_id` (str): `f"{subject_id}_{session_date}_{nwb_suffix}"`
- `data_source` (str): `"co_asset"`, `"bonsai_s3"`, `"bpod_s3"` — where trial data comes from
- `metadata_source` (str): `"han_pipeline"` or `"docdb_only"` (CO-only sessions absent from Han)
- `has_co_asset` (bool): True if session has a CO asset

**From `get_assets()` (supplement, CO sessions only)**:
- `code_ocean_asset_id`, `s3_nwb_location`, `modality`

### trial_table/ (AIND minimal canonical schema)

One row per trial. Partitioned by `subject_id` for fast per-mouse queries.

| Column | Type | Notes |
|--------|------|-------|
| `session_id` | str | join key to session_table |
| `subject_id` | str | partition key |
| `session_date` | str | YYYY-MM-DD |
| `nwb_suffix` | int | 0, 1, 2... or HHMMSS |
| `trial` | int | 1-based trial number |
| `animal_response` | int | 0=left, 1=right, 2=ignore |
| `goCue_start_time_in_session` | float | seconds from first goCue |
| `choice_time_in_session` | float | NaN if ignored |
| `reward_time_in_session` | float | NaN if not rewarded |
| `rewarded_historyL` | bool | reward baited/delivered left |
| `rewarded_historyR` | bool | reward baited/delivered right |
| `earned_reward` | bool | True if non-autowater reward |
| `reward_probabilityL` | float | scheduled p(reward) left |
| `reward_probabilityR` | float | scheduled p(reward) right |
| `data_source` | str | "co_asset", "bonsai_s3", "bpod_s3" |

**Column mapping for bonsai/bpod sessions**:
- `reward_non_autowater` (Han) → `earned_reward`
- `goCue_start_time` (Han, raw) → `goCue_start_time_in_session` = raw - t0 (first goCue)

### event_table/ (tidy events)

One row per event. Partitioned by `subject_id`.

| Column | Type | Notes |
|--------|------|-------|
| `session_id` | str | join key |
| `subject_id` | str | partition key |
| `session_date` | str | |
| `trial` | int | trial index (0-based, -1 = before first goCue) |
| `timestamps` | float | seconds from first goCue |
| `event` | str | event type name |
| `data` | float/str | event data value |
| `data_source` | str | |

---

## Builder Logic (`parquet_builder.py`)

### Session Table Build (full rebuild each run)
```python
def build_session_table() -> None:
    # Primary: han_pipeline
    df_han = get_session_table(if_load_bpod=True)
    df_han["session_id"] = df_han.apply(
        lambda r: f"{r.subject_id}_{r.session_date}_{r.nwb_suffix}", axis=1
    )

    # Supplement: docDB CO assets
    df_co = get_assets(subjects=[], processed=True)
    df_co["session_id"] = ...  # parse from session_name

    # Join: left join on session_id
    df = df_han.merge(df_co[["session_id", "code_ocean_asset_id", "has_co_asset"]],
                      on="session_id", how="left")
    df["metadata_source"] = "han_pipeline"

    df.to_parquet("s3://aind-behavior-data/foraging_cache/session_table.parquet")
```

### Trial/Event Table Build (incremental)
```python
def build_trial_and_event_tables(df_sessions, force_rebuild=False):
    processed = load_processed_sessions()  # from build_metadata.json

    for session in df_sessions.itertuples():
        if session.session_id in processed and not force_rebuild:
            continue

        if session.has_co_asset:
            nwb = load_nwb_from_filename(session.s3_nwb_location)
            df_trials = create_df_trials(nwb)       # nwb_utils.py
            df_events = create_df_events(nwb)       # nwb_utils.py
            source = "co_asset"

        elif session_in_bonsai(session):
            nwb_path = f"s3://aind-behavior-data/foraging_nwb_bonsai/{session.filename}"
            nwb = load_nwb_hdf5(nwb_path)
            df_trials = compute_df_trial_old(nwb)   # Han-provided, from process_nwbs.py
            df_events = compute_df_events_old(nwb)  # Han-provided
            df_trials = normalize_to_aind_schema(df_trials, source="bonsai_s3")
            source = "bonsai_s3"

        elif session_in_bpod(session):
            # Same logic, different S3 bucket
            source = "bpod_s3"
            ...

        # Add partition key, write
        df_trials["subject_id"] = session.subject_id
        write_parquet_partition(df_trials,
                                f"s3://aind-behavior-data/foraging_cache/trial_table/",
                                partition_cols=["subject_id"])
        processed.add(session.session_id)

    save_processed_sessions(processed)
```

### Schema Normalization
```python
def normalize_to_aind_schema(df_trials_han: pd.DataFrame, source: str) -> pd.DataFrame:
    t0 = df_trials_han["goCue_start_time"].iloc[0]  # first goCue
    df = pd.DataFrame()
    df["animal_response"] = df_trials_han["animal_response"]
    df["goCue_start_time_in_session"] = df_trials_han["goCue_start_time"] - t0
    df["rewarded_historyL"] = df_trials_han["rewarded_historyL"]
    df["rewarded_historyR"] = df_trials_han["rewarded_historyR"]
    df["earned_reward"] = df_trials_han["reward_non_autowater"]
    df["reward_probabilityL"] = df_trials_han["reward_probabilityL"]
    df["reward_probabilityR"] = df_trials_han["reward_probabilityR"]
    # choice_time and reward_time: compute from lick/reward timestamps
    df["data_source"] = source
    return df
```

---

## Query API (`cache_utils.py`)

```python
CACHE_BASE = "s3://aind-behavior-data/foraging_cache/"

def get_session_table(
    subjects: list = None,
    task: str = None,
    **kwargs
) -> pd.DataFrame:
    """Load full session table. Fast — single parquet file, ~MB scale."""
    df = pd.read_parquet(CACHE_BASE + "session_table.parquet")
    if subjects:
        df = df[df.subject_id.isin(subjects)]
    return df

def get_trial_table(
    subjects: list = None,
    session_ids: list = None,
    **kwargs
) -> pd.DataFrame:
    """Load trial table with predicate pushdown on subject_id partition.
    Querying one mouse reads only that partition — no full scan."""
    filters = []
    if subjects:
        filters.append(("subject_id", "in", subjects))
    if session_ids:
        filters.append(("session_id", "in", session_ids))
    return pd.read_parquet(CACHE_BASE + "trial_table/", filters=filters or None, engine="pyarrow")

def get_event_table(
    subjects: list = None,
    session_ids: list = None,
    event_types: list = None,
    **kwargs
) -> pd.DataFrame:
    """Load event table with predicate pushdown."""
    filters = []
    if subjects:
        filters.append(("subject_id", "in", subjects))
    if session_ids:
        filters.append(("session_id", "in", session_ids))
    return pd.read_parquet(CACHE_BASE + "event_table/", filters=filters or None, engine="pyarrow")
```

## Typical Usage Pattern

```python
from aind_dynamic_foraging_data_utils.cache_utils import get_session_table, get_trial_table

# Get all sessions for a cohort
df_sess = get_session_table(subjects=["712345", "712346", "712347"])

# Get trial data for those sessions
df_trials = get_trial_table(subjects=["712345", "712346", "712347"])

# Join session-level stats to trials
df = df_trials.merge(df_sess[["session_id", "foraging_eff", "PI"]], on="session_id")

# Filter and analyze
df_well_trained = df[df.foraging_eff > 0.6]
choice_history = df_well_trained.animal_response.map({0: "L", 1: "R", 2: "ignore"})
```
