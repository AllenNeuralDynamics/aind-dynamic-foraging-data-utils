"""
Builds and incrementally updates the foraging cache parquet tables on S3.

Three tables are built:
  1. session_table.parquet  - one row per session with all session-level metadata
  2. trial_table/           - Hive-partitioned by subject_id; one row per trial
  3. event_table/           - Hive-partitioned by subject_id; one row per behavioral event

S3 target: s3://aind-behavior-data/foraging_cache/

Data sources (priority order for NWB files):
  1. CO asset S3 URI        -- best; from docDB (AIND pipeline, ~2/3 of sessions)
  2. Local bonsai NWB       -- /data/foraging_nwb_bonsai/
  3. Local bpod NWB         -- /data/foraging_nwb_bpod/

NWB reader strategy (try-new-then-fallback):
  1. Try the AIND reader (nwb_utils.create_df_trials / create_df_events)
  2. On AINDReaderQualityError (assertion failures in post-2025 data):
     log a warning, fall back to the legacy Han pipeline reader
  3. For bpod NWBs: use the legacy reader directly (pynwb crashes on
     malformed bpod_backup_BehavioralEvents in many old bpod files)

Usage (small-scale test):
    from aind_dynamic_foraging_data_utils.foraging_cache import parquet_builder

    df_sess = parquet_builder.build_session_table(
        output_path="/tmp/foraging_cache/session_table.parquet",
        include_co_assets=False,
    )

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
      - Old bonsai/bpod : "{subject_id}_{date}_{HH-MM-SS}.nwb"
      - New bonsai      : "behavior_{subject_id}_{date}_{HH-MM-SS}.nwb"

    nwb_suffix is the time string with dashes removed as an int, e.g. "13-06-12" -> 130612.

    Returns:
        tuple (subject_id_str, session_date_str, nwb_suffix_int), or None if
        the filename does not match either pattern.
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
    Build a lookup dictionary from (subject_id, session_date, nwb_suffix) -> NWB filepath.

    Scans both bonsai and bpod directories. Bonsai files take priority over bpod
    when the same session key appears in both.

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
      1. Han pipeline session table  (primary metadata source)
      2. CO asset IDs from docDB     (for sessions with AIND-processed CO assets)
      3. Bowen incomplete session flags

    The resulting table has one row per session with all Han metadata plus:
      - co_asset_id       : CO data asset ID (str or None)
      - co_s3_nwb_uri     : S3 URI of the NWB inside the CO asset (str or None)
      - nwb_data_source   : "co_asset" | "bonsai_s3" | "bpod_s3"
      - is_bad_bowen_session : bool

    Returns:
        pd.DataFrame: The full session table (also written to output_path).
    """
    # ---- 1. Load Han pipeline session table (bonsai + bpod) ----
    if verbose:
        print("Loading Han session table (bonsai + bpod)...")

    from aind_analysis_arch_result_access.han_pipeline import get_session_table

    df_sessions = get_session_table(if_load_bpod=True)

    # Normalise nwb_suffix to nullable integer
    df_sessions["nwb_suffix"] = df_sessions["nwb_suffix"].astype("Int64")

    # Ensure subject_id is string
    df_sessions["subject_id"] = df_sessions["subject_id"].astype(str)

    # Normalise session_date to plain "YYYY-MM-DD" string
    df_sessions["session_date"] = pd.to_datetime(df_sessions["session_date"]).dt.strftime(
        "%Y-%m-%d"
    )

    if verbose:
        print(f"  Loaded {len(df_sessions)} sessions")

    # ---- 2. Initialise CO asset columns ----
    df_sessions["co_asset_id"] = pd.NA
    df_sessions["co_s3_nwb_uri"] = pd.NA

    if include_co_assets:
        if verbose:
            print("Querying docDB for CO asset metadata...")
        try:
            from aind_dynamic_foraging_data_utils.code_ocean_utils import add_s3_location, get_assets

            df_co = get_assets(processed=True, modality=["behavior"])

            if df_co is not None and len(df_co) > 0:
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

                if verbose:
                    print(f"  Found {len(df_co)} CO assets. Fetching S3 locations...")
                df_co = add_s3_location(df_co)

                co_lookup = {}
                for _, row in df_co.iterrows():
                    key = (row["_subject_id"], row["_session_date"], row["_nwb_suffix"])
                    co_lookup[key] = (
                        row.get("code_ocean_asset_id", pd.NA),
                        row.get("s3_nwb_location", pd.NA),
                    )

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
    def _assign_nwb_data_source(row):
        if pd.notna(row.get("co_asset_id")) and row["co_asset_id"] != "":
            return "co_asset"
        if "bpod" in str(row.get("data_source", "")).lower():
            return "bpod_s3"
        return "bonsai_s3"

    df_sessions["nwb_data_source"] = df_sessions.apply(_assign_nwb_data_source, axis=1)

    # ---- 5. Write to parquet ----
    if verbose:
        print(f"Writing session table ({len(df_sessions)} rows) -> {output_path} ...")

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
    """
    global _WORKER_NWB_INDEX, _WORKER_TRIAL_PREFIX, _WORKER_EVENT_PREFIX
    _WORKER_NWB_INDEX = nwb_file_index
    _WORKER_TRIAL_PREFIX = trial_prefix
    _WORKER_EVENT_PREFIX = event_prefix


def _process_single_session(row_dict):
    """
    Worker function: process one session and write its trial/event parquet files.

    NWB reader strategy:
      1. For bpod NWBs: use legacy reader directly (pynwb crashes on many old files)
      2. For bonsai / CO asset NWBs: try AIND reader first
      3. On AINDReaderQualityError: log warning, fall back to legacy reader

    Returns:
        dict: {
            "session_id" : str,
            "status"     : "ok" | "skipped" | "failed",
            "data_source": "co_asset" | "bonsai_s3" | "bpod_s3" | None,
            "reader"     : "aind" | "aind_fallback_legacy" | "legacy_bpod" | None,
            "error"      : str | None,
        }
    """
    from aind_dynamic_foraging_data_utils.foraging_cache import nwb_reader_aind, nwb_reader_legacy
    from aind_dynamic_foraging_data_utils.foraging_cache.nwb_reader_aind import (
        AINDReaderQualityError,
    )

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

    # ---- Determine NWB source (priority: CO asset > bonsai > bpod) ----
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
        return {
            "session_id": session_id,
            "status": "skipped",
            "data_source": None,
            "reader": None,
            "error": None,
        }

    # ---- Read NWB using try-AIND-then-fallback strategy ----
    try:
        df_trials, df_events, reader_used = _read_session_with_fallback(
            nwb_path, nwb_data_source, session_id
        )

        for df_out in [df_trials, df_events]:
            df_out["subject_id"] = subject_id
            df_out["session_date"] = session_date
            df_out["nwb_suffix"] = int(nwb_suffix) if pd.notna(nwb_suffix) else None
            df_out["session_id"] = session_id
            df_out["nwb_data_source"] = nwb_data_source

        _write_session_parquet(df_trials, trial_prefix, subject_id, session_id)
        _write_session_parquet(df_events, event_prefix, subject_id, session_id)

        return {
            "session_id": session_id,
            "status": "ok",
            "data_source": nwb_data_source,
            "reader": reader_used,
            "error": None,
        }

    except Exception as e:
        return {
            "session_id": session_id,
            "status": "failed",
            "data_source": nwb_data_source,
            "reader": None,
            "error": str(e),
        }


def _read_session_with_fallback(nwb_path, nwb_data_source, session_id):
    """
    Read trial and event tables from an NWB file using the try-AIND-then-legacy
    fallback strategy.

    For bpod NWBs: use legacy reader directly (pynwb crashes on many old files
    due to malformed bpod_backup_BehavioralEvents TimeSeries).

    For bonsai / CO asset NWBs:
      1. Try the AIND reader (nwb_utils.create_df_trials / create_df_events)
      2. On AINDReaderQualityError: log warning, fall back to legacy reader

    Returns:
        (df_trials, df_events, reader_used_str) where reader_used_str is one of:
          "aind"                -- data-util AIND reader, succeeded directly
          "aind_fallback_legacy"-- AIND reader failed, fell back to legacy bonsai reader
          "legacy_bpod"         -- legacy bpod reader used directly (no AIND attempted)
    """
    from aind_dynamic_foraging_data_utils.foraging_cache import nwb_reader_aind, nwb_reader_legacy
    from aind_dynamic_foraging_data_utils.foraging_cache.nwb_reader_aind import (
        AINDReaderQualityError,
    )

    # bpod NWBs -> legacy reader directly (pynwb can't even load many of them)
    if nwb_data_source == "bpod_s3":
        df_trials = nwb_reader_legacy.read_trials(nwb_path)
        df_events = nwb_reader_legacy.read_events(nwb_path)
        return df_trials, df_events, "legacy_bpod"

    # bonsai / CO asset -> try AIND first, fall back to legacy
    try:
        df_trials = nwb_reader_aind.read_trials(nwb_path)
        df_events = nwb_reader_aind.read_events(nwb_path)
        return df_trials, df_events, "aind"

    except AINDReaderQualityError as exc:
        logger.warning(
            "AIND reader quality error for %s: %s  -- falling back to legacy reader",
            session_id,
            exc,
        )
        df_trials = nwb_reader_legacy.read_trials(nwb_path)
        df_events = nwb_reader_legacy.read_events(nwb_path)
        return df_trials, df_events, "aind_fallback_legacy"


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

    NWB reader strategy per session:
      - bpod NWBs: legacy reader directly (avoids pynwb crash on old files)
      - bonsai / CO asset NWBs: try AIND reader first, fall back to legacy on
        AINDReaderQualityError (assertion failures in post-2025 data)

    Parallelism uses ProcessPoolExecutor because pynwb/h5py is not thread-safe.
    n_workers defaults to CO_CPUS env var, then os.cpu_count() - 1.

    Returns:
        dict: {
            "n_processed"          : int,
            "n_skipped"            : int,
            "n_failed"             : int,
            # data-source breakdown (processed sessions only)
            "n_co_asset"           : int,
            "n_bonsai_s3"          : int,
            "n_bpod_s3"            : int,
            # reader breakdown (processed sessions only)
            "n_aind_reader"        : int,   # data-util AIND reader, no fallback
            "n_aind_fallback_legacy": int,  # AIND tried, fell back to legacy bonsai
            "n_legacy_bpod"        : int,   # legacy bpod reader used directly
            "failed_sessions"      : list[dict],
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
            pass

    # ---- 2. Build list of sessions to process ----
    df = session_df.copy()

    df["_session_id"] = (
        df["subject_id"].astype(str)
        + "_"
        + df["session_date"].astype(str)
        + "_"
        + df["nwb_suffix"].astype(str)
    )

    if incremental and processed_sessions:
        df = df[~df["_session_id"].isin(processed_sessions)]

    if max_sessions is not None:
        df = df.head(max_sessions)

    n_total = len(df)
    if verbose:
        print(f"Processing {n_total} sessions (of {len(session_df)} total)...")

    # ---- 3. Resolve number of worker processes ----
    if n_workers is None:
        co_cpus = os.environ.get("CO_CPUS")
        if co_cpus is not None:
            n_workers = max(1, int(co_cpus) - 1)
        else:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

    if verbose:
        print(f"Using {n_workers} worker processes (CO_CPUS={os.environ.get('CO_CPUS', 'not set')})")

    # ---- 4. Submit tasks to ProcessPoolExecutor ----
    summary = {
        "n_processed": 0,
        "n_skipped": 0,
        "n_failed": 0,
        # data-source breakdown
        "n_co_asset": 0,
        "n_bonsai_s3": 0,
        "n_bpod_s3": 0,
        # reader breakdown
        "n_aind_reader": 0,
        "n_aind_fallback_legacy": 0,
        "n_legacy_bpod": 0,
        "failed_sessions": [],
    }
    newly_processed = []

    # Labels used in per-session log lines
    _READER_LABEL = {
        "aind": "aind",
        "aind_fallback_legacy": "aind->legacy",
        "legacy_bpod": "legacy_bpod",
    }
    _SOURCE_LABEL = {
        "co_asset": "CO-asset",
        "bonsai_s3": "bonsai-S3",
        "bpod_s3": "bpod-S3",
    }

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
            data_source = result.get("data_source")
            reader = result.get("reader")

            if verbose and (n_done <= 5 or n_done % 50 == 0):
                src_tag = f"  src={_SOURCE_LABEL.get(data_source, data_source)}" if data_source else ""
                rdr_tag = f"  rdr={_READER_LABEL.get(reader, reader)}" if reader else ""
                print(f"  [{n_done}/{n_total}] {session_id}  ->  {status}{src_tag}{rdr_tag}")

            if status == "ok":
                newly_processed.append(session_id)
                summary["n_processed"] += 1
                if data_source == "co_asset":
                    summary["n_co_asset"] += 1
                elif data_source == "bonsai_s3":
                    summary["n_bonsai_s3"] += 1
                elif data_source == "bpod_s3":
                    summary["n_bpod_s3"] += 1
                if reader == "aind":
                    summary["n_aind_reader"] += 1
                elif reader == "aind_fallback_legacy":
                    summary["n_aind_fallback_legacy"] += 1
                elif reader == "legacy_bpod":
                    summary["n_legacy_bpod"] += 1
            elif status == "skipped":
                summary["n_skipped"] += 1
            else:  # "failed"
                logger.warning(
                    "Failed to process %s  (src=%s): %s",
                    session_id, data_source, result["error"],
                )
                summary["n_failed"] += 1
                summary["failed_sessions"].append(
                    {
                        "session_id": session_id,
                        "data_source": data_source,
                        "error": result["error"],
                    }
                )

    # ---- 5. Update build metadata ----
    if build_metadata_path is not None and newly_processed:
        all_processed = list(processed_sessions | set(newly_processed))
        _write_json(
            {"processed_session_ids": all_processed, "n_processed": len(all_processed)},
            build_metadata_path,
        )

    if verbose:
        n_ok = summary["n_processed"]
        n_skip = summary["n_skipped"]
        n_fail = summary["n_failed"]
        print(
            f"\n{'='*50}\n"
            f"Build summary\n"
            f"{'='*50}\n"
            f"  Total submitted : {n_ok + n_skip + n_fail}\n"
            f"  Processed (ok)  : {n_ok}\n"
            f"  Skipped         : {n_skip}\n"
            f"  Failed          : {n_fail}\n"
            f"\n"
            f"  Data source breakdown (processed sessions):\n"
            f"    CO asset       : {summary['n_co_asset']}\n"
            f"    Bonsai S3      : {summary['n_bonsai_s3']}\n"
            f"    bpod S3        : {summary['n_bpod_s3']}\n"
            f"\n"
            f"  NWB reader breakdown (processed sessions):\n"
            f"    AIND data-util (direct)     : {summary['n_aind_reader']}\n"
            f"    AIND->legacy fallback        : {summary['n_aind_fallback_legacy']}\n"
            f"    Legacy bpod (direct)         : {summary['n_legacy_bpod']}\n"
            f"{'='*50}"
        )
        if summary["failed_sessions"]:
            print(f"\n  Failed sessions ({n_fail}):")
            for fs in summary["failed_sessions"][:20]:
                print(f"    [{fs.get('data_source', '?')}] {fs['session_id']}  --  {fs['error']}")
            if n_fail > 20:
                print(f"    ... and {n_fail - 20} more (see failed_sessions in return value)")

    return summary


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------


def _write_session_parquet(df, output_prefix, subject_id, session_id):
    """
    Write a single session's DataFrame to a Hive-partitioned parquet file.

    Output path: {output_prefix}/subject_id={subject_id}/{session_id}.parquet
    """
    df = df.copy()

    # Sanitise object columns that contain array-like values
    for col in df.select_dtypes(include="object").columns:
        sample_vals = df[col].dropna()
        if len(sample_vals) == 0:
            continue
        first = sample_vals.iloc[0]
        if hasattr(first, "__len__") and not isinstance(first, str):
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
    table = pa.Table.from_pandas(df, preserve_index=False)

    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        s3_path = path[len("s3://"):]
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
