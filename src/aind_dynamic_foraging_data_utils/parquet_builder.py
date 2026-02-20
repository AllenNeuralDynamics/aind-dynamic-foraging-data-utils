"""
Builds and incrementally updates the foraging cache parquet tables on S3.

Three tables are built:
  1. session_table.parquet  - one row per session with all session-level metadata
  2. trial_table/           - Hive-partitioned by subject_id; one row per trial (AIND minimal schema)
  3. event_table/           - Hive-partitioned by subject_id; one row per behavioral event

S3 target: s3://aind-behavior-data/foraging_cache/

Data sources (priority order for NWB files):
  1. CO asset S3 URI        -- best; from docDB (AIND pipeline, ~2/3 of sessions)
  2. Local bonsai NWB       -- /data/foraging_nwb_bonsai/ or s3://aind-behavior-data/foraging_nwb_bonsai/
  3. Local bpod NWB         -- /data/foraging_nwb_bpod/   or s3://aind-behavior-data/foraging_nwb_bpod/

Bad Bowen sessions (listed in Bowen_IncompleteSessions-081225.csv):
  - These sessions have unreliable bonsai data; must use CO asset instead.
  - Sessions flagged with is_bad_bowen_session=True in the output tables.

Usage (small-scale test):
    from aind_dynamic_foraging_data_utils import parquet_builder

    # Build session table (skip docDB for speed)
    df_sess = parquet_builder.build_session_table(
        output_path="/tmp/foraging_cache/session_table.parquet",
        include_co_assets=False,
    )

    # Build trial + event tables for the first 5 sessions
    index = parquet_builder.build_nwb_file_index()
    parquet_builder.build_trial_and_event_tables(
        session_df=df_sess,
        trial_output_prefix="/tmp/foraging_cache/trial_table",
        event_output_prefix="/tmp/foraging_cache/event_table",
        nwb_file_index=index,
        max_sessions=5,
    )
"""

import json
import logging
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from aind_dynamic_foraging_data_utils import nwb_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level globals used by worker processes.
# Set once per worker via _worker_init(); never touched by the main process.
# ---------------------------------------------------------------------------
_WORKER_NWB_INDEX = None
_WORKER_TRIAL_PREFIX = None
_WORKER_EVENT_PREFIX = None

# ---- Default S3 paths ----
S3_CACHE_BUCKET = "aind-behavior-data"
S3_CACHE_PREFIX = "foraging_cache"
SESSION_TABLE_S3_URI = f"s3://{S3_CACHE_BUCKET}/{S3_CACHE_PREFIX}/session_table.parquet"
TRIAL_TABLE_S3_PREFIX = f"s3://{S3_CACHE_BUCKET}/{S3_CACHE_PREFIX}/trial_table"
EVENT_TABLE_S3_PREFIX = f"s3://{S3_CACHE_BUCKET}/{S3_CACHE_PREFIX}/event_table"
BUILD_METADATA_S3_URI = f"s3://{S3_CACHE_BUCKET}/{S3_CACHE_PREFIX}/build_metadata.json"

# ---- Local NWB paths (when running inside Code Ocean) ----
LOCAL_BONSAI_NWB_DIR = "/data/foraging_nwb_bonsai"
LOCAL_BPOD_NWB_DIR = "/data/foraging_nwb_bpod"

# ---- CSV with Bowen incomplete sessions (CO asset IDs) ----
BOWEN_INCOMPLETE_CSV = "/data/Bowen_IncompleteSessions-081225.csv"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _parse_nwb_filename(filename):
    """
    Parse a bonsai/bpod NWB filename to extract (subject_id, session_date, nwb_suffix).

    Handles two filename formats:
      - Old bonsai/bpod : "{subject_id}_{date}_{HH-MM-SS}.nwb"   e.g. "688237_2023-08-11_11-47-36.nwb"
      - New bonsai      : "behavior_{subject_id}_{date}_{HH-MM-SS}.nwb"  e.g. "behavior_771432_2024-12-16_13-06-12.nwb"

    nwb_suffix is the time string with dashes removed as an int, e.g. "13-06-12" → 130612.
    This matches the nwb_suffix column in get_session_table().

    Returns:
        tuple (subject_id_str, session_date_str, nwb_suffix_int), or None if the filename
        does not match either pattern.
    """
    fname = os.path.basename(filename)
    if not fname.endswith(".nwb"):
        return None

    stem = fname[:-4]  # strip .nwb

    # New format: behavior_{subject_id}_{YYYY-MM-DD}_{HH-MM-SS}
    m = re.match(r"behavior_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$", stem)
    if m:
        subject_id, session_date, time_str = m.groups()
        return subject_id, session_date, int(time_str.replace("-", ""))

    # Old format: {subject_id}_{YYYY-MM-DD}_{HH-MM-SS}
    m = re.match(r"(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$", stem)
    if m:
        subject_id, session_date, time_str = m.groups()
        return subject_id, session_date, int(time_str.replace("-", ""))

    return None


def build_nwb_file_index(
    bonsai_dir=LOCAL_BONSAI_NWB_DIR,
    bpod_dir=LOCAL_BPOD_NWB_DIR,
):
    """
    Build a lookup dictionary from (subject_id, session_date, nwb_suffix) → NWB filepath.

    Scans both bonsai and bpod directories. Bonsai files take priority over bpod
    when the same session key appears in both (bonsai files overwrite bpod entries).
    Non-NWB files (logs, json, csv) in the directories are silently skipped.

    Args:
        bonsai_dir (str): Path to the directory containing bonsai NWB files.
        bpod_dir   (str): Path to the directory containing bpod NWB files.

    Returns:
        dict: { (subject_id_str, session_date_str, nwb_suffix_int): filepath_str }
    """
    index = {}

    # Index bpod first (lower priority; bonsai entries will overwrite below)
    if os.path.isdir(bpod_dir):
        for fname in os.listdir(bpod_dir):
            parsed = _parse_nwb_filename(fname)
            if parsed is None:
                continue
            index[parsed] = os.path.join(bpod_dir, fname)

    # Index bonsai second (higher priority; overwrites bpod on collision)
    if os.path.isdir(bonsai_dir):
        for fname in os.listdir(bonsai_dir):
            parsed = _parse_nwb_filename(fname)
            if parsed is None:
                continue
            index[parsed] = os.path.join(bonsai_dir, fname)

    logger.info("NWB file index built: %d sessions found", len(index))
    return index


def _load_bowen_incomplete_sessions(csv_path=BOWEN_INCOMPLETE_CSV):
    """
    Load the set of CO asset IDs corresponding to Bowen's incomplete sessions.

    These sessions have unreliable bonsai data; the parquet builder will only
    use their CO asset S3 NWB and will flag them with is_bad_bowen_session=True.

    Args:
        csv_path (str): Path to the CSV file (one CO asset ID per line, no header).

    Returns:
        set: CO asset IDs (str). Empty set if the file is not found.
    """
    if not os.path.exists(csv_path):
        warnings.warn(f"Bowen incomplete sessions CSV not found: {csv_path}")
        return set()

    df = pd.read_csv(csv_path, header=None, names=["co_asset_id"])
    return set(df["co_asset_id"].str.strip())


# ---------------------------------------------------------------------------
# Session table builder
# ---------------------------------------------------------------------------


def build_session_table(
    output_path=SESSION_TABLE_S3_URI,
    bowen_csv_path=BOWEN_INCOMPLETE_CSV,
    include_co_assets=True,
    verbose=True,
):
    """
    Build the session-level parquet table by combining:
      1. Han pipeline session table  (primary metadata source; includes bpod sessions)
      2. CO asset IDs from docDB     (for sessions with AIND-processed CO assets)
      3. Bowen incomplete session flags

    The resulting table has one row per session with all Han session metadata plus:
      - co_asset_id       : CO data asset ID (str or None)
      - co_s3_nwb_uri     : S3 URI of the NWB inside the CO asset (str or None)
      - nwb_data_source   : "co_asset" | "bonsai_s3" | "bpod_s3" — preferred NWB source
      - is_bad_bowen_session : bool — True for Bowen sessions with unreliable bonsai data

    Note: Han's session table already has a 'data_source' column that records the
    rig/pipeline origin (e.g. "AIND_training_446_bonsai"). The new 'nwb_data_source'
    column is distinct and indicates which NWB file to use for trial/event extraction.

    Args:
        output_path (str)     : Local path or S3 URI (s3://...) for the output parquet.
        bowen_csv_path (str)  : Path to the Bowen incomplete sessions CSV.
        include_co_assets (bool): If True, query docDB to enrich with CO asset IDs and S3
                                  locations. Set to False for fast test runs (skips docDB).
        verbose (bool)        : Print progress messages.

    Returns:
        pd.DataFrame: The full session table (also written to output_path).
    """
    # ---- 1. Load Han pipeline session table (bonsai + bpod) ----
    if verbose:
        print("Loading Han session table (bonsai + bpod)...")

    from aind_analysis_arch_result_access.han_pipeline import get_session_table

    df_sessions = get_session_table(if_load_bpod=True)

    # Normalise nwb_suffix to nullable integer (source gives float for some rows)
    df_sessions["nwb_suffix"] = df_sessions["nwb_suffix"].astype("Int64")

    # Ensure subject_id is string for consistent joining
    df_sessions["subject_id"] = df_sessions["subject_id"].astype(str)

    # Normalise session_date to plain "YYYY-MM-DD" string so that parquet round-trips
    # don't silently convert it to datetime64 (which breaks key lookups).
    df_sessions["session_date"] = pd.to_datetime(df_sessions["session_date"]).dt.strftime(
        "%Y-%m-%d"
    )

    if verbose:
        print(f"  Loaded {len(df_sessions)} sessions")

    # ---- 2. Initialise CO asset columns (filled below if include_co_assets=True) ----
    df_sessions["co_asset_id"] = pd.NA
    df_sessions["co_s3_nwb_uri"] = pd.NA

    if include_co_assets:
        if verbose:
            print("Querying docDB for CO asset metadata...")
        try:
            from aind_dynamic_foraging_data_utils.code_ocean_utils import add_s3_location, get_assets

            df_co = get_assets(processed=True, modality=["behavior"])

            if df_co is not None and len(df_co) > 0:
                # Parse (subject_id, session_date, nwb_suffix) from CO asset session_name.
                # session_name format: "behavior_{subject_id}_{YYYY-MM-DD}_{HH-MM-SS}"
                def _parse_co_session_name(sname):
                    m = re.match(
                        r"behavior_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$",
                        str(sname),
                    )
                    if m:
                        subj, date, time_str = m.groups()
                        return subj, date, int(time_str.replace("-", ""))
                    return None, None, None

                parsed = df_co["session_name"].apply(_parse_co_session_name)
                df_co = df_co.copy()
                df_co["_subject_id"] = parsed.apply(lambda x: x[0])
                df_co["_session_date"] = parsed.apply(lambda x: x[1])
                df_co["_nwb_suffix"] = parsed.apply(lambda x: x[2]).astype("Int64")
                df_co = df_co.dropna(subset=["_subject_id"])

                # Fetch S3 NWB locations
                if verbose:
                    print(f"  Found {len(df_co)} CO assets. Fetching S3 locations...")
                df_co = add_s3_location(df_co)

                # Build lookup: (subject_id, session_date, nwb_suffix) → (co_asset_id, s3_uri)
                co_lookup = {}
                for _, row in df_co.iterrows():
                    key = (row["_subject_id"], row["_session_date"], row["_nwb_suffix"])
                    co_lookup[key] = (
                        row.get("code_ocean_asset_id", pd.NA),
                        row.get("s3_nwb_location", pd.NA),
                    )

                # Join CO asset info into session table
                def _lookup(r, field):
                    val = co_lookup.get(
                        (str(r["subject_id"]), str(r["session_date"]), r["nwb_suffix"]),
                        (pd.NA, pd.NA),
                    )
                    return val[0] if field == "id" else val[1]

                df_sessions["co_asset_id"] = df_sessions.apply(
                    lambda r: _lookup(r, "id"), axis=1
                )
                df_sessions["co_s3_nwb_uri"] = df_sessions.apply(
                    lambda r: _lookup(r, "uri"), axis=1
                )

                n_matched = df_sessions["co_asset_id"].notna().sum()
                if verbose:
                    print(f"  Matched {n_matched}/{len(df_sessions)} sessions with CO assets")

        except Exception as e:
            warnings.warn(f"Failed to load CO assets from docDB: {e}")

    # ---- 3. Flag Bowen incomplete sessions ----
    bowen_bad_co_ids = _load_bowen_incomplete_sessions(bowen_csv_path)
    df_sessions["is_bad_bowen_session"] = df_sessions["co_asset_id"].isin(bowen_bad_co_ids)

    if verbose:
        print(f"  Flagged {df_sessions['is_bad_bowen_session'].sum()} Bowen incomplete sessions")

    # ---- 4. Assign preferred NWB data source per session ----
    # CO asset is best. For sessions without CO asset, infer bonsai vs bpod from
    # Han's 'data_source' column (contains rig description like "..._bpod").
    def _assign_nwb_data_source(row):
        if pd.notna(row.get("co_asset_id")) and row["co_asset_id"] != "":
            return "co_asset"
        if "bpod" in str(row.get("data_source", "")).lower():
            return "bpod_s3"
        return "bonsai_s3"

    df_sessions["nwb_data_source"] = df_sessions.apply(_assign_nwb_data_source, axis=1)

    # ---- 5. Write to parquet ----
    if verbose:
        print(f"Writing session table ({len(df_sessions)} rows) → {output_path} ...")

    _write_dataframe_as_parquet(df_sessions, output_path)

    if verbose:
        print("  Done!")

    return df_sessions


# ---------------------------------------------------------------------------
# Parallel worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------


def _worker_init(nwb_file_index, trial_prefix, event_prefix):
    """
    Initializer for ProcessPoolExecutor workers.

    Called once when each worker process starts. Stores shared data as
    module-level globals so they don't need to be re-pickled for every task.

    Args:
        nwb_file_index (dict): (subject_id, date, nwb_suffix) → NWB filepath.
        trial_prefix   (str) : Output prefix for trial parquet files.
        event_prefix   (str) : Output prefix for event parquet files.
    """
    global _WORKER_NWB_INDEX, _WORKER_TRIAL_PREFIX, _WORKER_EVENT_PREFIX
    _WORKER_NWB_INDEX = nwb_file_index
    _WORKER_TRIAL_PREFIX = trial_prefix
    _WORKER_EVENT_PREFIX = event_prefix


def _process_single_session(row_dict):
    """
    Worker function: process one session and write its trial/event parquet files.

    Reads an NWB file (CO asset → bonsai → bpod priority), runs
    create_df_trials() and create_df_events(), annotates the rows with session
    identity, then writes Hive-partitioned parquet.

    This is a top-level function (not a closure or lambda) so that Python's
    multiprocessing can pickle it. Shared data (NWB index, output prefixes) are
    accessed via module-level globals set by _worker_init().

    Args:
        row_dict (dict): One row from the session DataFrame serialised as a dict.

    Returns:
        dict: {
            "session_id": str,
            "status"    : "ok" | "skipped" | "failed",
            "error"     : str | None,
        }
    """
    nwb_file_index = _WORKER_NWB_INDEX
    trial_prefix = _WORKER_TRIAL_PREFIX
    event_prefix = _WORKER_EVENT_PREFIX

    subject_id = str(row_dict["subject_id"])

    # Normalise session_date to plain "YYYY-MM-DD" string
    raw_date = row_dict["session_date"]
    if hasattr(raw_date, "strftime"):
        session_date = raw_date.strftime("%Y-%m-%d")
    else:
        session_date = str(raw_date)[:10]

    nwb_suffix = row_dict["nwb_suffix"]
    session_id = row_dict["_session_id"]
    is_bad_bowen = bool(row_dict.get("is_bad_bowen_session", False))
    co_s3_uri = row_dict.get("co_s3_nwb_uri", None)

    # ---- Determine NWB source (same priority as serial version) ----
    nwb_path = None
    nwb_data_source = None

    if pd.notna(co_s3_uri) and co_s3_uri != "":
        nwb_path = co_s3_uri
        nwb_data_source = "co_asset"
    elif not is_bad_bowen:
        suffix_int = int(nwb_suffix) if pd.notna(nwb_suffix) else -1
        key = (subject_id, session_date, suffix_int)
        if key in nwb_file_index:
            nwb_path = nwb_file_index[key]
            nwb_data_source = (
                "bpod_s3" if nwb_path.startswith(LOCAL_BPOD_NWB_DIR) else "bonsai_s3"
            )

    if nwb_path is None:
        return {"session_id": session_id, "status": "skipped", "error": None}

    # ---- Read NWB and write parquet ----
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_trials = nwb_utils.create_df_trials(nwb_path, verbose=False)
            df_events = nwb_utils.create_df_events(nwb_path, verbose=False)

        for df_out in [df_trials, df_events]:
            df_out["subject_id"] = subject_id
            df_out["session_date"] = session_date
            df_out["nwb_suffix"] = int(nwb_suffix) if pd.notna(nwb_suffix) else None
            df_out["session_id"] = session_id
            df_out["nwb_data_source"] = nwb_data_source

        _write_session_parquet(df_trials, trial_prefix, subject_id, session_id)
        _write_session_parquet(df_events, event_prefix, subject_id, session_id)

        return {"session_id": session_id, "status": "ok", "error": None}

    except Exception as e:
        return {"session_id": session_id, "status": "failed", "error": str(e)}


# ---------------------------------------------------------------------------
# Trial / event table builder
# ---------------------------------------------------------------------------


def build_trial_and_event_tables(
    session_df,
    trial_output_prefix,
    event_output_prefix,
    nwb_file_index=None,
    build_metadata_path=None,
    incremental=True,
    max_sessions=None,
    n_workers=None,
    verbose=True,
):
    """
    Build trial-level and event-level Hive-partitioned parquet tables from NWB files,
    using parallel processing across multiple CPU cores.

    For each session in session_df the function:
      1. Determines the NWB file path using priority:
           a. co_s3_nwb_uri  — CO asset S3 URI (best quality)
           b. Local bonsai file from nwb_file_index
           c. Local bpod file from nwb_file_index
         Bad Bowen sessions (is_bad_bowen_session=True) are skipped if no CO asset
         URI is available (their bonsai data is unreliable).
      2. Reads the NWB and runs:
           - nwb_utils.create_df_trials()  → AIND minimal trial schema
           - nwb_utils.create_df_events()  → tidy event table
      3. Annotates each row with subject_id, session_date, nwb_suffix, session_id,
         and nwb_data_source.
      4. Writes to Hive-partitioned parquet:
           {trial_output_prefix}/subject_id={subject_id}/{session_id}.parquet
           {event_output_prefix}/subject_id={subject_id}/{session_id}.parquet

    Parallelism:
      Uses ProcessPoolExecutor (multiprocessing, not threading) because:
        - pynwb / h5py is not thread-safe for parallel HDF5 reads
        - numpy operations in create_df_trials() are CPU-bound (GIL matters)
        - Each session is independent — no shared write paths
      n_workers defaults to the CO_CPUS environment variable (set by Code Ocean),
      falling back to os.cpu_count() - 1.

    Incremental updates:
      Previously processed sessions are tracked in build_metadata.json.
      Only new sessions are processed when incremental=True.

    Args:
        session_df (pd.DataFrame)   : Session table from build_session_table().
        trial_output_prefix (str)   : Local dir or S3 prefix for trial parquet.
        event_output_prefix (str)   : Local dir or S3 prefix for event parquet.
        nwb_file_index (dict)       : Pre-built index from build_nwb_file_index().
                                      Built automatically (local dirs) if None.
        build_metadata_path (str)   : Path / S3 URI for build_metadata.json used to
                                      track processed sessions. None disables tracking.
        incremental (bool)          : If True, skip already-processed sessions.
        max_sessions (int)          : Process at most this many sessions (useful for tests).
        n_workers (int | None)      : Number of parallel worker processes. Defaults to
                                      CO_CPUS env var, then os.cpu_count() - 1.
        verbose (bool)              : Print progress.

    Returns:
        dict: {
            "n_processed": int,   # sessions successfully written
            "n_skipped":   int,   # sessions with no NWB file found
            "n_failed":    int,   # sessions where NWB read/write failed
            "failed_sessions": list[dict],  # [{"session_id": ..., "error": ...}]
        }
    """
    # ---- 0. Build NWB file index if not provided ----
    if nwb_file_index is None:
        if verbose:
            print("Building NWB file index from local directories...")
        nwb_file_index = build_nwb_file_index()
        if verbose:
            print(f"  {len(nwb_file_index)} NWB files indexed")

    # ---- 1. Load incremental build metadata ----
    processed_sessions = set()
    if incremental and build_metadata_path is not None:
        try:
            metadata = _read_json(build_metadata_path)
            processed_sessions = set(metadata.get("processed_session_ids", []))
            if verbose:
                print(f"Incremental mode: {len(processed_sessions)} sessions already processed")
        except (FileNotFoundError, KeyError):
            pass  # First run; metadata file doesn't exist yet

    # ---- 2. Build list of sessions to process ----
    df = session_df.copy()

    # Synthesise a unique session ID string for tracking
    df["_session_id"] = (
        df["subject_id"].astype(str)
        + "_"
        + df["session_date"].astype(str)
        + "_"
        + df["nwb_suffix"].astype(str)
    )

    # Skip already-processed sessions in incremental mode
    if incremental and processed_sessions:
        df = df[~df["_session_id"].isin(processed_sessions)]

    if max_sessions is not None:
        df = df.head(max_sessions)

    n_total = len(df)
    if verbose:
        print(f"Processing {n_total} sessions (of {len(session_df)} total)...")

    # ---- 3. Resolve number of worker processes ----
    # CO_CPUS is set by Code Ocean to the number of CPUs allocated to the capsule.
    if n_workers is None:
        co_cpus = os.environ.get("CO_CPUS")
        if co_cpus is not None:
            n_workers = max(1, int(co_cpus) - 1)  # leave 1 core for the main process
        else:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

    if verbose:
        print(f"Using {n_workers} worker processes (CO_CPUS={os.environ.get('CO_CPUS', 'not set')})")

    # ---- 4. Submit tasks to ProcessPoolExecutor ----
    # _worker_init sets the NWB index and output prefixes as module-level globals
    # in each worker process (once on startup), so they don't need to be pickled
    # with every individual task.
    summary = {"n_processed": 0, "n_skipped": 0, "n_failed": 0, "failed_sessions": []}
    newly_processed = []

    row_dicts = [row.to_dict() for _, row in df.iterrows()]

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(nwb_file_index, trial_output_prefix, event_output_prefix),
    ) as executor:
        futures = {executor.submit(_process_single_session, rd): rd["_session_id"]
                   for rd in row_dicts}

        for n_done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            session_id = result["session_id"]
            status = result["status"]

            # Progress logging: first 5 completions, then every 50
            if verbose and (n_done <= 5 or n_done % 50 == 0):
                print(f"  [{n_done}/{n_total}] {session_id}  →  {status}")

            if status == "ok":
                newly_processed.append(session_id)
                summary["n_processed"] += 1
            elif status == "skipped":
                summary["n_skipped"] += 1
            else:  # "failed"
                logger.warning("Failed to process %s: %s", session_id, result["error"])
                summary["n_failed"] += 1
                summary["failed_sessions"].append(
                    {"session_id": session_id, "error": result["error"]}
                )

    # ---- 5. Update build metadata ----
    if build_metadata_path is not None and newly_processed:
        all_processed = list(processed_sessions | set(newly_processed))
        _write_json(
            {"processed_session_ids": all_processed, "n_processed": len(all_processed)},
            build_metadata_path,
        )

    if verbose:
        print(
            f"\nDone! Processed: {summary['n_processed']}, "
            f"Skipped: {summary['n_skipped']}, "
            f"Failed: {summary['n_failed']}"
        )

    return summary


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------


def _write_session_parquet(df, output_prefix, subject_id, session_id):
    """
    Write a single session's DataFrame to a Hive-partitioned parquet file.

    Output path: {output_prefix}/subject_id={subject_id}/{session_id}.parquet

    Object-typed columns that contain arrays (e.g. lists of lick timestamps)
    are converted to their string representation so pyarrow can serialise them.
    """
    df = df.copy()

    # Sanitise any remaining object columns that contain array-like values
    for col in df.select_dtypes(include="object").columns:
        sample_vals = df[col].dropna()
        if len(sample_vals) == 0:
            continue
        first = sample_vals.iloc[0]
        if hasattr(first, "__len__") and not isinstance(first, str):
            # Convert arrays to their string representation for parquet compatibility
            df[col] = df[col].apply(
                lambda x: str(list(x)) if hasattr(x, "__len__") and not isinstance(x, str) else x
            )

    table = pa.Table.from_pandas(df, preserve_index=False)
    partition_dir = f"subject_id={subject_id}"

    if output_prefix.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        s3_base = output_prefix[len("s3://"):]
        out_path = f"{s3_base}/{partition_dir}/{session_id}.parquet"
        with fs.open(out_path, "wb") as f:
            pq.write_table(table, f)
    else:
        out_dir = os.path.join(output_prefix, partition_dir)
        os.makedirs(out_dir, exist_ok=True)
        pq.write_table(table, os.path.join(out_dir, f"{session_id}.parquet"))


def _write_dataframe_as_parquet(df, path):
    """Write a DataFrame as a single parquet file to a local path or S3 URI."""
    # Convert all-NA columns to object to avoid pyarrow type inference issues
    table = pa.Table.from_pandas(df, preserve_index=False)

    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        s3_path = path[len("s3://"):]
        # Ensure parent prefix exists (S3 doesn't need mkdir, but local does)
        with fs.open(s3_path, "wb") as f:
            pq.write_table(table, f)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pq.write_table(table, path)


def _write_json(data, path):
    """Write a dict as JSON to a local path or S3 URI."""
    json_str = json.dumps(data, indent=2)
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path[len("s3://"):], "w") as f:
            f.write(json_str)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(json_str)


def _read_json(path):
    """Read a JSON file from a local path or S3 URI and return as dict."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path[len("s3://"):], "r") as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)
