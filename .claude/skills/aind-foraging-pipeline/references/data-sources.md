# Data Sources: AIND Dynamic Foraging

## Source 1: AIND CO Assets (New NWB / zarr format)

**Coverage**: ~2/3 of all sessions (AIND pipeline has not backfilled all historical data)
**Format**: NWB zarr (directory-based) or HDF5 `.nwb`
**Location**:
- Attached CO asset: `/data/{asset_name}/nwb/{session_name}.nwb`
- S3: discovered via `add_s3_location()` or docDB `location` field

**How to access**:
```python
from aind_dynamic_foraging_data_utils import code_ocean_utils, nwb_utils

# Query docDB for CO assets
results = code_ocean_utils.get_assets(subjects=[], processed=True, modality=["behavior"])
# results has columns: name, session_name, code_ocean_asset_id, location, ...

# Load NWB
nwb = nwb_utils.load_nwb_from_filename("/data/{asset_name}/nwb/{session_name}.nwb")
df_trials = nwb_utils.create_df_trials(nwb)
df_events = nwb_utils.create_df_events(nwb)
```

**Session ID format in NWB**: `behavior_{subject_id}_{date}_{HHMMSS}` (new) or `{subject_id}_{date}[_{n}].json` (old)
**Quirks**: Some sessions have duplicate docDB entries (deduped by processing time in `get_assets()`); zarr format requires `NWBZarrIO`

---

## Source 2: Han Pipeline — Bonsai S3 (Old NWB / HDF5)

**Coverage**: Essentially all sessions from Bonsai-Harp hardware at AIND
**Format**: Old HDF5 NWB (not zarr).
**Location**: `s3://aind-behavior-data/foraging_nwb_bonsai/`
**Filename pattern**: `{subject_id}_{date}_{time}.nwb` or `{subject_id}_{date}[_{n}].nwb`

**How to access (trial data)**:
- Use Han-provided `compute_df_trial(nwb)` from `aind-foraging-behavior-bonsai-basic/code/process_nwbs.py`
- Then map to AIND canonical schema: rename `reward_non_autowater` → `earned_reward`, compute `*_in_session` timestamps by subtracting first goCue time
- Load via: `NWBHDF5IO(s3_path, mode='r')` with s3fs filesystem

**Bpod detection**: `'bpod' in nwb.session_description` — if True, foraging_eff_random_seed is pre-computed in metadata

---

## Source 3: Han Pipeline — Bpod S3 (Old NWB / HDF5)

**Coverage**: Older sessions from bpod hardware (pre-Bonsai-Harp)
**Format**: Same old HDF5 NWB format as bonsai but with `'bpod' in nwb.session_description`
**Location**: `s3://aind-behavior-data/foraging_nwb_bpod/`

**How to access**: Same as bonsai source — `compute_df_trial(nwb)` handles bpod automatically via the `'bpod' in session_description` flag.

---

## Source 4: Han Pipeline Session Table (Primary Metadata)

**Coverage**: All sessions (bonsai + bpod, aggregated)
**What it contains**: Per-session stats: `foraging_eff`, `foraging_eff_random_seed`, `finished_trials`, `finished_rate`, `bias_naive`, lick stats (means/medians), duration stats, curriculum info, weight/water metrics, rig info

**How to access**:
```python
from aind_analysis_arch_result_access.han_pipeline import get_session_table
df_sessions = get_session_table(if_load_bpod=True)
# Returns flat DataFrame (hierarchical columns already flattened)
# Key columns: subject_id, session_date, nwb_suffix, session, rig, trainer, PI,
#              curriculum_name, current_stage_actual, finished_trials, foraging_eff,
#              finished_rate, bias_naive, weight_after, water_in_session_total, task, notes
```

**Data flow**: Reads `df_sessions.pkl` from `S3_PATH_BONSAI_ROOT` (the aggregated output of the bonsai-basic pipeline), enriches with docDB curriculum data and mouse-PI mapping.

---

## Source 5: docDB (AIND Metadata Database)

**Coverage**: ~2/3 of sessions (same as CO assets)
**What it contains**: CO asset IDs, S3 locations, rig, modality, experimenter, task type, training stage, session metadata in AIND format

**How to access**:
```python
from aind_dynamic_foraging_data_utils.code_ocean_utils import get_assets
results = get_assets(
    subjects=[],              # empty = all subjects (slow)
    processed=True,
    task=[],                  # empty = all baiting variants
    modality=["behavior"],
    stage=[],
    extra_filter={},
)
# Key columns: name, session_name, code_ocean_asset_id, external_links, location, session, subject
```

**Database**: MongoDB at `api.allenneuraldynamics.org`, collection `metadata_index.data_assets`
**Session name format**: `behavior_{subject_id}_{date}_processed_...` (name field)

---

## Data Source Priority for Parquet Builder

```
For each session:
  1. Has CO asset? → nwb_utils.create_df_trials() [AIND canonical schema]
  2. In bonsai S3? → compute_df_trial() + normalize to AIND schema
  3. In bpod S3?  → compute_df_trial() + normalize to AIND schema
  4. None found   → log as missing, skip
```

Record `data_source` column in every row: `"co_asset"`, `"bonsai_s3"`, or `"bpod_s3"`.

---

## Session Join Key Design

The join between `get_session_table()` and `get_assets()` uses `(subject_id, session_date)` as the base key, with `nwb_suffix` for disambiguation when multiple sessions exist per day.

- `nwb_suffix` is an integer: 0 (or small int for old format), or HHMMSS integer for new format
- han_pipeline session table has `nwb_suffix` column
- docDB session name encodes this in the filename: `behavior_{subject}_{date}_{HHMMSS}_processed_...`
- For CO sessions with multiple per day: docDB is deduplicated by keeping latest processing run
