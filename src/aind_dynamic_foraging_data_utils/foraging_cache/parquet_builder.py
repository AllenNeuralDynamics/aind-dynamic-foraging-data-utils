"""
Builds and incrementally updates the foraging cache parquet tables on S3.

Three tables are built:
  1. session_table.parquet  - one row per session with all session-level metadata
  2. trial_table/           - Hive-partitioned by subject_id; one row per trial
  3. event_table/           - Hive-partitioned by subject_id; one row per behavioral event

S3 target: s3://aind-scratch-data/aind-dynamic-foraging-cache/

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
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
S3_CACHE_BUCKET = "aind-scratch-data"
S3_CACHE_PREFIX = "aind-dynamic-foraging-cache"
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
# Canonical trial-table column definitions
# ---------------------------------------------------------------------------
#
# The trial table is produced by three different readers:
#   1. AIND reader  (nwb_utils.create_df_trials)  — CO asset + modern bonsai
#   2. Legacy bonsai reader                        — older bonsai NWBs
#   3. Legacy bpod reader                          — bpod NWBs
#
# Readers 2 and 3 produce:
#   • different time-column names  (e.g. bare "goCue_start_time" vs AIND's
#     "goCue_start_time_in_session")
#   • 22 "bpod_backup_*" columns that are raw hardware backup data with no
#     behavioral meaning and are NOT present in the AIND reader output
#
# Policy: we keep the AIND / CO-asset column set as canonical and drop
# bpod_backup_* columns on write so they never pollute the parquet schema.
#
# Type-normalisation map: resolves the remaining cross-session type conflicts
# (e.g. bool in one session, string or double in another).
# Resolution rule — highest-priority type wins:
#   string  >  double  >  int64  >  bool
# (string can represent any value; double absorbs NaN; int64 > int32)

_TRIAL_COLS_TO_DROP_PREFIX = ("bpod_backup_",)

_CANONICAL_TRIAL_COL_TYPES: dict = {
    # bool vs string (can contain 'none') → string
    "auto_train_engaged": "string",
    "auto_train_stage_overridden": "string",
    "session_wide_control": "string",
    # bool vs double (no string; True/False → 1.0/0.0; NaN-safe) → double
    "bait_left": "double",
    "bait_right": "double",
    "laser_on_trial": "double",
    # int32 vs int64 → int64
    "auto_waterL": "int64",
    "auto_waterR": "int64",
    "non_autowater_trial": "int64",
    # double vs int64 (no string) → double
    "laser_condition_probability": "double",
    "laser_duration": "double",
    "laser_end_offset": "double",
    "laser_frequency": "double",
    "laser_power": "double",
    "laser_pulse_duration": "double",
    "laser_rampingdown": "double",
    "laser_start_offset": "double",
    "laser_wavelength": "double",
    # double/int/string (ever string) → string
    "laser_condition": "string",
    "laser_end": "string",
    "laser_location": "string",
    "laser_protocol": "string",
    "laser_start": "string",
    "left_reward_type": "string",
    "right_reward_type": "string",
}

# Event table: only the 'data' column conflicts (double vs string → string)
_CANONICAL_EVENT_COL_TYPES: dict = {
    "data": "string",
}


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
# Session table helpers (module-level for reuse and pickling)
# ---------------------------------------------------------------------------


def _parse_co_session_name(sname):
    """Parse a CO session_name like 'behavior_123456_2024-01-01_13-06-12'
    into (subject_id, session_date, nwb_suffix_int), or (None, None, None)."""
    m = re.match(
        r"behavior_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$",
        str(sname),
    )
    if m:
        subj, date, time_str = m.groups()
        return subj, date, int(time_str.replace("-", ""))
    return None, None, None


def _assign_nwb_data_source(row):
    """Determine the preferred NWB data source for a session row."""
    if pd.notna(row.get("co_asset_id")) and row["co_asset_id"] != "":
        return "co_asset"
    if "bpod" in str(row.get("data_source", "")).lower():
        return "bpod_s3"
    return "bonsai_s3"


def _compute_session_id(df):
    """Return a Series of '{subject_id}_{session_date}_{nwb_suffix}' strings."""
    return (
        df["subject_id"].astype(str)
        + "_"
        + df["session_date"].astype(str)
        + "_"
        + df["nwb_suffix"].astype(str)
    )


def _read_existing_session_table(path):
    """Read a session table parquet from local or S3 path.

    Returns pd.DataFrame or None if file doesn't exist.
    """
    try:
        if path.startswith("s3://"):
            fs = s3fs.S3FileSystem(anon=False)
            s3_path = path[len("s3://") :]
            if not fs.exists(s3_path):
                return None
            return pd.read_parquet(path, filesystem=fs)
        else:
            if not os.path.exists(path):
                return None
            return pd.read_parquet(path)
    except Exception:
        return None


def _add_s3_location_parallel(df, n_workers=100, verbose=True):
    """Parallel replacement for code_ocean_utils.add_s3_location().

    Resolves each CO asset's ``.nwb`` S3 path. The NWB lives at a deterministic
    ``{location}/nwb/{session_name}.nwb``, so we construct that path and confirm
    it with a single (non-recursive) ``fs.exists`` call — cheap. Only if that
    misses (e.g. zarr-directory NWBs or atypical layouts) do we fall back to a
    recursive ``{location}/**/*.nwb`` glob, which lists every object under the
    asset. This keeps the common case to one S3 call instead of a full listing.

    Uses a ThreadPoolExecutor since the work is I/O-bound (S3 API calls). More
    threads do not scale past s3fs's shared connection pool, so ~100 is plenty.
    """
    fs = s3fs.S3FileSystem()

    def _resolve_one(location, session_name):
        """Return the s3:// path of the asset's .nwb, or '' if none found."""
        # Fast path: deterministic constructed path, verified with one exists().
        if isinstance(session_name, str) and session_name:
            candidate = f"{location}/nwb/{session_name}.nwb"
            try:
                if fs.exists(candidate):
                    return candidate if candidate.startswith("s3://") else f"s3://{candidate}"
            except Exception:
                pass
        # Fallback: recursive glob (handles zarr dirs / atypical layouts).
        try:
            hits = fs.glob(f"{location}/**/*.nwb")
            return f"s3://{hits[0]}" if hits else ""
        except Exception:
            return ""

    locations = df["location"].tolist()
    if "session_name" in df.columns:
        session_names = df["session_name"].tolist()
    else:
        session_names = [None] * len(locations)

    n_actual = min(n_workers, len(locations)) if locations else 1
    if verbose:
        print(f"    Using {n_actual} threads to resolve {len(locations)} NWB S3 paths")

    results = [""] * len(locations)
    with ThreadPoolExecutor(max_workers=max(1, n_actual)) as pool:
        future_to_idx = {
            pool.submit(_resolve_one, loc, sn): i
            for i, (loc, sn) in enumerate(zip(locations, session_names))
        }
        for n_done, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            results[idx] = future.result()
            if verbose and (n_done <= 3 or n_done % 50 == 0 or n_done == len(locations)):
                print(f"    NWB path resolve: {n_done}/{len(locations)}")

    df = df.copy()
    df["s3_nwb_location"] = results
    return df


def _get_assets_with_retry(chunk, max_retries=4, base_delay=2):
    """
    Call get_assets() for one subject chunk, retrying transient docDB errors.

    docDB (api.allenneuraldynamics.org) intermittently returns 503 Service
    Unavailable under load. Without retries a single 503 in any chunk aborts the
    whole enrichment (leaving every session with no CO asset). Here we retry with
    exponential backoff and, if a chunk still fails, return None so the caller can
    skip just those subjects — they get picked up on the next incremental run via
    recheck_missing_co.
    """
    from aind_dynamic_foraging_data_utils.code_ocean_utils import get_assets

    last_exc = None
    for attempt in range(max_retries):
        try:
            return get_assets(subjects=chunk, processed=True, modality=["behavior"])
        except Exception as e:  # transient docDB errors (503, timeouts, ...)
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    logger.warning(
        "docDB query failed for %d subjects after %d retries (skipped this run): %s",
        len(chunk),
        max_retries,
        last_exc,
    )
    return None


def _enrich_with_co_assets(df_sessions, subject_ids, verbose=True):  # noqa: C901
    """Query docDB for CO assets for the given subject_ids and merge onto df_sessions.

    Populates 'co_asset_id' and 'co_s3_nwb_uri' columns via targeted
    get_assets(subjects=subject_ids) + add_s3_location().

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Must have subject_id, session_date, nwb_suffix columns.
    subject_ids : list[str]
        Subject IDs to query. If empty, skips the query entirely.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        df_sessions with co_asset_id and co_s3_nwb_uri updated in place.
    """
    if not subject_ids:
        if verbose:
            print("  No subjects to query for CO assets — skipping.")
        return df_sessions

    if verbose:
        print(f"  Querying docDB for CO assets (subjects={len(subject_ids)})...")

    # Chunk subjects and query docDB in parallel threads to avoid a single
    # massive regex and to overlap network latency.
    CHUNK_SIZE = 10
    chunks = [subject_ids[i : i + CHUNK_SIZE] for i in range(0, len(subject_ids), CHUNK_SIZE)]
    n_threads = min(len(chunks), 20)

    if verbose:
        print(
            f"    Using {n_threads} threads for {len(chunks)} docDB query chunks "
            f"({CHUNK_SIZE} subjects/chunk)"
        )

    co_frames = []
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(_get_assets_with_retry, chunk): i for i, chunk in enumerate(chunks)
        }
        for n_done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None and len(result) > 0:
                co_frames.append(result)
            if verbose and (n_done <= 3 or n_done % 10 == 0 or n_done == len(chunks)):
                print(f"    docDB query: {n_done}/{len(chunks)} chunks done")

    df_co = pd.concat(co_frames, ignore_index=True) if co_frames else None

    if df_co is None or len(df_co) == 0:
        if verbose:
            print("  No CO assets found for these subjects.")
        return df_sessions

    parsed = df_co["session_name"].apply(_parse_co_session_name)
    df_co = df_co.copy()
    df_co["_subject_id"] = parsed.apply(lambda x: x[0])
    df_co["_session_date"] = parsed.apply(lambda x: x[1])
    df_co["_nwb_suffix"] = parsed.apply(lambda x: x[2]).astype("Int64")
    df_co = df_co.dropna(subset=["_subject_id"])

    if verbose:
        print(f"  Found {len(df_co)} CO assets from docDB (all sessions for queried subjects)")

    # Filter to only CO assets that match sessions in df_sessions, so we
    # don't waste time S3-globbing thousands of irrelevant assets.
    session_keys = set(
        zip(
            df_sessions["subject_id"].astype(str),
            df_sessions["session_date"].astype(str),
            df_sessions["nwb_suffix"],
        )
    )
    match_mask = [
        (row["_subject_id"], row["_session_date"], row["_nwb_suffix"]) in session_keys
        for _, row in df_co.iterrows()
    ]
    df_co = df_co[match_mask]

    if verbose:
        print(f"  Narrowed to {len(df_co)} CO assets matching requested sessions")

    if len(df_co) == 0:
        return df_sessions

    if verbose:
        print("  Fetching S3 locations (parallel)...")
    df_co = _add_s3_location_parallel(df_co, verbose=verbose)

    co_lookup = {}
    for _, row in df_co.iterrows():
        key = (row["_subject_id"], row["_session_date"], row["_nwb_suffix"])
        co_lookup[key] = (
            row.get("code_ocean_asset_id", pd.NA),
            row.get("s3_nwb_location", pd.NA),
        )

    def _lookup(r, field):
        """Return co_asset_id ('id') or co_s3_nwb_uri ('uri') for row r."""
        val = co_lookup.get(
            (str(r["subject_id"]), str(r["session_date"]), r["nwb_suffix"]),
            (pd.NA, pd.NA),
        )
        return val[0] if field == "id" else val[1]

    df_sessions["co_asset_id"] = df_sessions.apply(lambda r: _lookup(r, "id"), axis=1)
    df_sessions["co_s3_nwb_uri"] = df_sessions.apply(lambda r: _lookup(r, "uri"), axis=1)

    n_matched = df_sessions["co_asset_id"].notna().sum()
    if verbose:
        print(f"  Matched {n_matched}/{len(df_sessions)} sessions with CO assets")

    return df_sessions


# ---------------------------------------------------------------------------
# Session table builder
# ---------------------------------------------------------------------------


def _merge_han_and_co(df_han, df_co, verbose=True):  # noqa: C901
    """
    Merge the Han session universe with the docDB / Code Ocean (CO) universe into
    one complete session table, attaching CO assets per the agreed match rule.

    Why two universes: Han's pipeline is still running but is slated to be shut
    down eventually, and it did not capture some sessions that exist on Code Ocean.
    The docDB discovery (get_dynamic_foraging_assets) is the complete CO universe.
    We union them so the cache covers every session (and keeps working once Han's
    pipeline is retired).

    Match rule (see GitHub issue #146 for the multi-session-per-day background):

      Han has at most ONE session per (subject_id, session_date) — verified — which
      is what makes the 2-tuple safe for the single-session case. Han's pipeline
      enforces this itself: when a mouse has multiple NWBs on one day, its
      add_session_number() keeps only the session with the most finished_trials
      (assigns it the session number) and sets the others to NaN. So Han's "one
      session that day" is specifically the max-finished-trials one — which is also
      why its HH-MM-SS lines up with one of the CO sessions in the 3-tuple match.
      See aind-foraging-behavior-bonsai-basic/code/process_nwbs.py#L569-L573:
      https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/blob/0b63a460682b8b497be5b9c14329e4520c92ffda/code/process_nwbs.py#L569-L573

      * CO has ONE session that (subject, date)  [single-session-per-day]:
          - (subject, date) present in Han  -> attach the CO asset to that Han row
            via the 2-tuple. This deliberately tolerates the common HH-MM-SS drift
            between Han's nwb_suffix and docDB's (same session, different recorded
            time), which exact-suffix matching otherwise misses.
          - (subject, date) absent from Han -> add as a NEW CO-only row; we are
            sure it is the only real session that day.

      * CO has MULTIPLE sessions that (subject, date)  [multi-session-per-day]:
          - if exactly one of them matches a Han session by the 3-tuple
            (subject, date, nwb_suffix) -> attach that one to the Han row.
            (Verified: such days never have >1 exact match -> unambiguous.)
          - every other same-day CO session, and ALL sessions on multi-session
            days with no exact match, are NOT merged or added — they are returned
            in ``df_skipped`` with a reason. We cannot safely tell which CO session
            Han refers to, so we log rather than guess (issue #146).

    The deterministic NWB S3 path is constructed from the asset root + session_name
    (``{location}/nwb/{session_name}.nwb``) — no per-asset S3 glob needed.

    Parameters
    ----------
    df_han : pd.DataFrame
        Han sessions; must have subject_id (str), session_date ("YYYY-MM-DD"),
        nwb_suffix (Int64), _session_id, plus Han metadata columns.
    df_co : pd.DataFrame
        docDB discovery from get_dynamic_foraging_assets(): session_name, location,
        code_ocean_asset_id, subject_id.
    verbose : bool

    Returns
    -------
    (df_union, df_skipped) : tuple[pd.DataFrame, pd.DataFrame]
        df_union   - df_han with co_asset_id / co_s3_nwb_uri filled where matched,
                     plus appended CO-only NEW rows (Han metadata columns are NaN).
        df_skipped - multi-session CO rows that were logged and NOT used, with a
                     ``skip_reason`` column (for triage; see issue #146).
    """
    # --- Parse the CO universe into (subject, date, suffix) + constructed URI ---
    co = df_co.copy()
    parsed = co["session_name"].str.extract(
        r"^behavior_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$"
    )
    co["subject_id"] = parsed[0]
    co["session_date"] = parsed[1]
    co["nwb_suffix"] = parsed[2].str.replace("-", "", regex=False)
    co = co.dropna(subset=["subject_id", "session_date", "nwb_suffix"]).copy()
    co["nwb_suffix"] = co["nwb_suffix"].astype(int)
    co["co_s3_nwb_uri"] = (
        co["location"].astype(str) + "/nwb/" + co["session_name"].astype(str) + ".nwb"
    )
    co = co.rename(columns={"code_ocean_asset_id": "co_asset_id"})

    # single- vs multi-session days within the CO universe
    day_size = co.groupby(["subject_id", "session_date"])["nwb_suffix"].transform("size")
    co["_is_single_day"] = day_size == 1
    co["_in_han2"] = [
        (s, d) in set(zip(df_han["subject_id"], df_han["session_date"]))
        for s, d in zip(co["subject_id"], co["session_date"])
    ]

    # --- Lookups for attaching CO -> Han ---
    co3 = {  # exact 3-tuple -> (asset_id, uri), for ALL CO sessions
        (r.subject_id, r.session_date, int(r.nwb_suffix)): (r.co_asset_id, r.co_s3_nwb_uri)
        for r in co.itertuples(index=False)
    }
    co_single2 = {  # 2-tuple -> (asset_id, uri), SINGLE-day CO sessions only
        (r.subject_id, r.session_date): (r.co_asset_id, r.co_s3_nwb_uri)
        for r in co[co["_is_single_day"]].itertuples(index=False)
    }
    han3 = {
        (s, d, int(x))
        for s, d, x in zip(
            df_han["subject_id"],
            df_han["session_date"],
            [v if pd.notna(v) else -1 for v in df_han["nwb_suffix"]],
        )
    }

    # --- Attach CO to each Han row: exact 3-tuple first, then single-day 2-tuple ---
    def _co_for_han(row):
        suf = int(row["nwb_suffix"]) if pd.notna(row["nwb_suffix"]) else -1
        hit = co3.get((row["subject_id"], row["session_date"], suf))
        if hit is None:
            hit = co_single2.get((row["subject_id"], row["session_date"]))
        return hit if hit is not None else (pd.NA, pd.NA)

    df_han = df_han.copy()
    pairs = df_han.apply(_co_for_han, axis=1)
    df_han["co_asset_id"] = [p[0] for p in pairs]
    df_han["co_s3_nwb_uri"] = [p[1] for p in pairs]

    # --- CO-only NEW rows: single-day CO sessions whose (subject,date) not in Han ---
    co_new = co[co["_is_single_day"] & ~co["_in_han2"]]
    new_rows = pd.DataFrame(
        {
            "subject_id": co_new["subject_id"].to_numpy(),
            "session_date": co_new["session_date"].to_numpy(),
            "nwb_suffix": pd.array(co_new["nwb_suffix"].to_numpy(), dtype="Int64"),
            "co_asset_id": co_new["co_asset_id"].to_numpy(),
            "co_s3_nwb_uri": co_new["co_s3_nwb_uri"].to_numpy(),
        }
    )
    if len(new_rows):
        new_rows["_session_id"] = _compute_session_id(new_rows)

    # --- Skipped multi-session CO sessions (logged, not used) ---
    multi = co[~co["_is_single_day"]].copy()
    multi["_exact"] = [
        (r.subject_id, r.session_date, int(r.nwb_suffix)) in han3
        for r in multi.itertuples(index=False)
    ]
    matched_days = set(
        multi.loc[multi["_exact"], ["subject_id", "session_date"]].itertuples(index=False, name=None)
    )
    df_skipped = multi[~multi["_exact"]].copy()
    df_skipped["skip_reason"] = [
        "extra same-day session (Han matched another suffix)"
        if (r.subject_id, r.session_date) in matched_days
        else "multi-session day, no Han exact match (ambiguous)"
        for r in df_skipped.itertuples(index=False)
    ]

    df_union = pd.concat([df_han, new_rows], ignore_index=True)

    if verbose:
        n_attached = int(df_han["co_asset_id"].notna().sum())
        print(
            f"  CO merge: {n_attached} Han sessions matched a CO asset, "
            f"{len(new_rows)} CO-only sessions added, "
            f"{len(df_skipped)} multi-session CO rows skipped (see skip log)."
        )

    return df_union, df_skipped


def build_session_table(  # noqa: C901
    output_path=SESSION_TABLE_S3_URI,
    bowen_csv_path=BOWEN_INCOMPLETE_CSV,
    include_co_assets=True,
    co_discovery=None,
    verbose=True,
):
    """
    Build the complete session-level parquet table by unioning two universes:
      1. Han pipeline session table  — rich behavioural metadata (foraging_eff,
         finished_trials, curriculum, ...); at most one session per (subject, date).
      2. docDB / Code Ocean universe — every processed dynamic-foraging session
         (get_dynamic_foraging_assets), catching sessions missing from Han and
         attaching the CO asset + constructed S3 NWB path.

    CO assets are attached / unioned via _merge_han_and_co() — see its docstring
    and GitHub issue #146 for the single- vs multi-session-per-day match rule
    (single-day: 2-tuple match/add; multi-day: keep only the exact 3-tuple match,
    log+skip the rest). This replaces the old per-subject docDB enrichment + S3
    globbing: one paginated discovery query, then deterministic path construction.

    The resulting table has one row per session with all Han metadata (NaN for
    CO-only sessions Han never tracked) plus:
      - _session_id          : unique key "{subject_id}_{session_date}_{nwb_suffix}"
      - co_asset_id          : CO data asset ID (str or NA)
      - co_s3_nwb_uri        : S3 URI of the NWB inside the CO asset (str or NA)
      - nwb_data_source      : "co_asset" | "bonsai_s3" | "bpod_s3"
      - is_bad_bowen_session : bool

    Parameters
    ----------
    output_path : str
        Where to write the parquet (local path or S3 URI). A companion
        ``co_skipped_sessions.csv`` (the skipped multi-session CO rows, issue #146)
        is written alongside it.
    bowen_csv_path : str
        Path to the Bowen incomplete-sessions CSV.
    include_co_assets : bool
        If False, skip the docDB discovery (Han-only table; co columns are NA).
    co_discovery : pd.DataFrame | None
        Optional pre-fetched get_dynamic_foraging_assets() result, to skip the
        ~137 s docDB query (handy for dev iteration / local caching). None -> fetch.
    verbose : bool

    Returns:
        pd.DataFrame: the full session table (also written to output_path).
    """
    # ---- 1. Load Han pipeline session table (bonsai + bpod) ----
    if verbose:
        print("Loading Han session table (bonsai + bpod)...")

    from aind_analysis_arch_result_access.han_pipeline import get_session_table

    df_han = get_session_table(if_load_bpod=True)

    # Normalise nwb_suffix to nullable integer
    df_han["nwb_suffix"] = df_han["nwb_suffix"].astype("Int64")

    # Ensure subject_id is string
    df_han["subject_id"] = df_han["subject_id"].astype(str)

    # Normalise session_date to plain "YYYY-MM-DD" string
    df_han["session_date"] = pd.to_datetime(df_han["session_date"]).dt.strftime("%Y-%m-%d")

    # Compute stable session ID
    df_han["_session_id"] = _compute_session_id(df_han)

    if verbose:
        print(f"  Loaded {len(df_han)} sessions from Han table")

    # ---- 2. Union with the docDB / CO universe (or build Han-only) ----
    df_skipped = None
    if include_co_assets:
        if co_discovery is None:
            from aind_dynamic_foraging_data_utils.code_ocean_utils import (
                get_dynamic_foraging_assets,
            )

            if verbose:
                print("Querying docDB for the complete dynamic-foraging CO universe...")
            co_discovery = get_dynamic_foraging_assets()
        df_sessions, df_skipped = _merge_han_and_co(df_han, co_discovery, verbose=verbose)
    else:
        df_sessions = df_han.copy()
        df_sessions["co_asset_id"] = pd.NA
        df_sessions["co_s3_nwb_uri"] = pd.NA

    # ---- 7. Flag Bowen incomplete sessions (refresh on all rows) ----
    bowen_bad_co_ids = _load_bowen_incomplete_sessions(bowen_csv_path)
    df_sessions["is_bad_bowen_session"] = df_sessions["co_asset_id"].isin(bowen_bad_co_ids)

    if verbose:
        print(f"  Flagged {df_sessions['is_bad_bowen_session'].sum()} Bowen incomplete sessions")

    # ---- 8. Assign preferred NWB data source per session (refresh on all rows) ----
    df_sessions["nwb_data_source"] = df_sessions.apply(_assign_nwb_data_source, axis=1)

    # ---- 9. Write to parquet ----
    if verbose:
        print(f"Writing session table ({len(df_sessions)} rows) -> {output_path} ...")

    _write_dataframe_as_parquet(df_sessions, output_path)

    # Companion log of the skipped multi-session CO rows (issue #146), for triage.
    if df_skipped is not None and len(df_skipped):
        skipped_path = (
            output_path.rsplit("/", 1)[0] + "/co_skipped_sessions.csv"
            if "/" in output_path
            else "co_skipped_sessions.csv"
        )
        cols = [c for c in ["session_name", "subject_id", "co_asset_id", "skip_reason"]
                if c in df_skipped.columns]
        _write_csv(df_skipped[cols], skipped_path)
        if verbose:
            print(f"  Skipped {len(df_skipped)} multi-session CO rows -> {skipped_path}")

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

    NWB reader strategy (see references/data-sources.md):
      - CO asset  -> AIND reader (nwb_utils) on the docDB S3 URI. On
        AINDReaderQualityError, fall back to the legacy reader on the local Han
        NWB (if one exists for this session).
      - bonsai S3 -> legacy reader directly (the AIND reader is NOT run on Han
        bonsai NWBs).
      - bpod S3   -> legacy reader directly.

    Returns:
        dict: {
            "session_id" : str,
            "status"     : "ok" | "skipped" | "failed",
            "data_source": "co_asset" | "bonsai_s3" | "bpod_s3" | None,
            "reader"     : "aind" | "aind_fallback_legacy"
                           | "legacy_bonsai" | "legacy_bpod" | None,
            "error"      : str | None,
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

    # ---- Determine NWB source (priority: CO asset > bonsai > bpod) ----
    # The local Han bonsai/bpod NWB for this session (if any). It is the primary
    # path for bonsai/bpod sessions, and the legacy-reader fallback when the AIND
    # reader fails on a CO asset.
    suffix_int = int(nwb_suffix) if pd.notna(nwb_suffix) else -1
    local_nwb_path = nwb_file_index.get((subject_id, session_date, suffix_int))

    nwb_path = None
    nwb_data_source = None
    legacy_fallback_path = None

    if pd.notna(co_s3_uri) and co_s3_uri != "":
        # CO asset -> AIND reader on the S3 URI; fall back to the local Han NWB.
        nwb_path = co_s3_uri
        nwb_data_source = "co_asset"
        legacy_fallback_path = local_nwb_path
    elif not is_bad_bowen and local_nwb_path is not None:
        # No CO asset -> read the local Han NWB directly with the legacy reader.
        nwb_path = local_nwb_path
        nwb_data_source = (
            "bpod_s3" if local_nwb_path.startswith(LOCAL_BPOD_NWB_DIR) else "bonsai_s3"
        )

    # Common identity fields recorded for every session (for the triage log).
    base = {
        "session_id": session_id,
        "subject_id": subject_id,
        "session_date": session_date,
        "nwb_suffix": int(nwb_suffix) if pd.notna(nwb_suffix) else None,
    }

    if nwb_path is None:
        return {
            **base,
            "status": "skipped",
            "data_source": None,
            "reader": None,
            "nwb_path": None,
            "n_trials": 0,
            "n_events": 0,
            "error": "no CO asset and no local NWB (bad-Bowen or missing file)",
        }

    # ---- Read NWB (AIND reader for CO assets, legacy reader for Han NWBs) ----
    try:
        df_trials, df_events, reader_used = _read_session_with_fallback(
            nwb_path, nwb_data_source, session_id, legacy_fallback_path
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
            **base,
            "status": "ok",
            "data_source": nwb_data_source,
            "reader": reader_used,
            "nwb_path": nwb_path,
            "n_trials": len(df_trials),
            "n_events": len(df_events),
            "error": None,
        }

    except Exception as e:
        return {
            **base,
            "status": "failed",
            "data_source": nwb_data_source,
            "reader": None,
            "nwb_path": nwb_path,
            "n_trials": 0,
            "n_events": 0,
            "error": str(e),
        }


def _read_session_with_fallback(nwb_path, nwb_data_source, session_id, legacy_fallback_path=None):
    """
    Route a session to the correct NWB reader.

    The AIND data-util reader (nwb_utils.create_df_trials / create_df_events) is
    designed for AIND-pipeline CO-asset NWBs and is used ONLY for those. Han
    bonsai/bpod NWBs are read with the legacy reader directly.

      - co_asset  : AIND reader on the CO-asset S3 URI. On ANY error that breaks
                    the AIND reader — quality assertions (AINDReaderQualityError,
                    e.g. post-2025 "Reward before choice time") as well as
                    TypeErrors / parse / S3 failures — log the reason and fall
                    back to the legacy reader on the local Han NWB
                    (legacy_fallback_path), if available.
      - bonsai_s3 : legacy reader directly.
      - bpod_s3   : legacy reader directly (pynwb also crashes on many old bpod
                    files; the legacy reader has an h5py fallback).

    Args:
        nwb_path (str)            : Primary NWB path (CO S3 URI, or local Han NWB).
        nwb_data_source (str)     : "co_asset" | "bonsai_s3" | "bpod_s3".
        session_id (str)          : For log messages.
        legacy_fallback_path (str): Local Han NWB to use if the AIND reader fails
                                    on a CO asset. None disables the fallback.

    Returns:
        (df_trials, df_events, reader_used_str) where reader_used_str is one of:
          "aind"                 -- AIND reader on the CO asset, succeeded
          "aind_fallback_legacy" -- AIND reader failed; legacy reader on Han NWB
          "legacy_bonsai"        -- legacy reader on a Han bonsai NWB
          "legacy_bpod"          -- legacy reader on a Han bpod NWB
    """
    from aind_dynamic_foraging_data_utils.foraging_cache import nwb_reader_aind, nwb_reader_legacy
    from aind_dynamic_foraging_data_utils.foraging_cache.nwb_reader_aind import (
        AINDReaderQualityError,
    )

    # Han bonsai/bpod NWBs -> legacy reader directly (never the AIND reader).
    if nwb_data_source == "bpod_s3":
        return (
            nwb_reader_legacy.read_trials(nwb_path),
            nwb_reader_legacy.read_events(nwb_path),
            "legacy_bpod",
        )
    if nwb_data_source == "bonsai_s3":
        return (
            nwb_reader_legacy.read_trials(nwb_path),
            nwb_reader_legacy.read_events(nwb_path),
            "legacy_bonsai",
        )

    # CO asset -> AIND reader on the S3 URI; fall back to the local Han NWB.
    try:
        return (
            nwb_reader_aind.read_trials(nwb_path),
            nwb_reader_aind.read_events(nwb_path),
            "aind",
        )
    except Exception as exc:
        # Fall back on ANY AIND-reader breakage: quality assertions
        # (AINDReaderQualityError) as well as TypeErrors, parse/S3 errors, etc.
        # Distinguish the kind in the log so quality rejections stay visible.
        kind = "quality" if isinstance(exc, AINDReaderQualityError) else type(exc).__name__
        if legacy_fallback_path is None:
            logger.warning(
                "AIND reader failed for %s (%s): %s -- no local Han NWB to fall back to",
                session_id,
                kind,
                exc,
            )
            raise
        logger.warning(
            "AIND reader failed for %s (%s): %s -- falling back to legacy reader",
            session_id,
            kind,
            exc,
        )
        return (
            nwb_reader_legacy.read_trials(legacy_fallback_path),
            nwb_reader_legacy.read_events(legacy_fallback_path),
            "aind_fallback_legacy",
        )


# ---------------------------------------------------------------------------
# Trial / event table builder
# ---------------------------------------------------------------------------


def build_trial_and_event_tables(  # noqa: C901
    session_df,
    trial_output_prefix,
    event_output_prefix,
    nwb_file_index=None,
    build_metadata_path=None,
    incremental=True,
    max_sessions=None,
    n_workers=None,
    coalesce=True,
    log_csv_path=None,
    verbose=True,
):
    """
    Build trial-level and event-level Hive-partitioned parquet tables from NWB files,
    using parallel processing across multiple CPU cores.

    NWB reader strategy per session (see references/data-sources.md):
      - CO asset  : AIND reader on the docDB S3 URI; on AINDReaderQualityError,
        fall back to the legacy reader on the local Han NWB.
      - bonsai S3 : legacy reader directly (AIND reader is NOT used on Han NWBs).
      - bpod S3   : legacy reader directly.

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
            "n_aind_reader"        : int,   # AIND reader on CO asset, no fallback
            "n_aind_fallback_legacy": int,  # AIND tried on CO asset, fell back to legacy
            "n_legacy_bonsai"      : int,   # legacy reader on Han bonsai NWB
            "n_legacy_bpod"        : int,   # legacy reader on Han bpod NWB
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

    df["_session_id"] = _compute_session_id(df)

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
        print(
            f"Using {n_workers} worker processes (CO_CPUS={os.environ.get('CO_CPUS', 'not set')})"
        )

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
        "n_legacy_bonsai": 0,
        "n_legacy_bpod": 0,
        "failed_sessions": [],
    }
    newly_processed = []
    all_results = []  # every session's result row, for the triage CSV log
    affected_subjects = set()  # subjects with newly-written sessions, for coalescing

    # Labels used in per-session log lines
    _READER_LABEL = {
        "aind": "aind",
        "aind_fallback_legacy": "aind->legacy",
        "legacy_bonsai": "legacy_bonsai",
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
        futures = {
            executor.submit(_process_single_session, rd): rd["_session_id"] for rd in row_dicts
        }

        for n_done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            all_results.append(result)
            session_id = result["session_id"]
            status = result["status"]
            data_source = result.get("data_source")
            reader = result.get("reader")

            if verbose and (n_done <= 5 or n_done % 50 == 0):
                src_tag = (
                    f"  src={_SOURCE_LABEL.get(data_source, data_source)}" if data_source else ""
                )
                rdr_tag = f"  rdr={_READER_LABEL.get(reader, reader)}" if reader else ""
                print(f"  [{n_done}/{n_total}] {session_id}  ->  {status}{src_tag}{rdr_tag}")

            if status == "ok":
                newly_processed.append(session_id)
                affected_subjects.add(str(result["subject_id"]))
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
                elif reader == "legacy_bonsai":
                    summary["n_legacy_bonsai"] += 1
                elif reader == "legacy_bpod":
                    summary["n_legacy_bpod"] += 1
            elif status == "skipped":
                summary["n_skipped"] += 1
            else:  # "failed"
                logger.warning(
                    "Failed to process %s  (src=%s): %s",
                    session_id,
                    data_source,
                    result["error"],
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

    # ---- 6. Append to the human-readable triage CSV log ----
    if log_csv_path is not None and all_results:
        _append_triage_log(log_csv_path, all_results, verbose=verbose)

    # ---- 7. Coalesce each affected subject's per-session files into one ----
    if coalesce and affected_subjects:
        if verbose:
            print(f"\nCoalescing {len(affected_subjects)} subject partitions into one file each...")
        _coalesce_partitions(
            trial_output_prefix, affected_subjects,
            sort_cols=["session_date", "nwb_suffix", "trial"], verbose=verbose,
        )
        _coalesce_partitions(
            event_output_prefix, affected_subjects,
            sort_cols=["session_date", "nwb_suffix", "timestamps"], verbose=verbose,
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
# Triage log + coalescing
# ---------------------------------------------------------------------------

# Columns of the human-readable per-session triage log (one row per session).
_TRIAGE_COLUMNS = [
    "session_id", "subject_id", "session_date", "nwb_suffix",
    "status", "data_source", "reader", "n_trials", "n_events",
    "nwb_path", "error", "processed_at",
]


def _append_triage_log(log_csv_path, results, verbose=True):
    """
    Write/merge a human-readable CSV with one row per processed session.

    Each row records the triage detail (status, data source, reader used, row
    counts, NWB path, error). Existing rows are merged in and de-duplicated by
    session_id (the latest run wins), so the file is a cumulative state of every
    session ever processed.
    """
    import datetime as _dt

    now = _dt.datetime.now().isoformat(timespec="seconds")
    new_rows = [{**r, "processed_at": now} for r in results]
    df_new = pd.DataFrame(new_rows)
    df_new = df_new.reindex(columns=_TRIAGE_COLUMNS)

    # Merge with any existing log (latest row per session_id wins).
    try:
        df_old = _read_csv(log_csv_path)
        df_old = df_old[~df_old["session_id"].isin(set(df_new["session_id"]))]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all = df_all.sort_values(["subject_id", "session_date", "nwb_suffix"]).reset_index(drop=True)
    _write_csv(df_all, log_csv_path)
    if verbose:
        print(f"  Triage log: {len(df_new)} sessions updated -> {log_csv_path} ({len(df_all)} total)")


def _coalesce_partitions(output_prefix, subject_ids, sort_cols, n_threads=16, verbose=True):
    """Coalesce each subject's per-session parquet files into one sorted file."""
    subject_ids = list(subject_ids)
    n = min(len(subject_ids), n_threads)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
        futures = {
            pool.submit(_coalesce_subject, output_prefix, s, sort_cols): s for s in subject_ids
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # don't let one subject abort the rest
                logger.warning("Coalesce failed for subject %s: %s", futures[fut], e)
            done += 1
            if verbose and (done % 100 == 0 or done == len(subject_ids)):
                print(f"    coalesced {done}/{len(subject_ids)} subjects")


def _coalesce_subject(output_prefix, subject_id, sort_cols):
    """
    Merge all per-session parquet files in subject_id=<id>/ into one sorted file
    named {subject_id}.parquet.

    Sessions already present in a prior coalesced file are replaced by any newly
    written per-session files for the same session_id (so incremental adds and
    reprocessing both stay correct). The per-session files are removed afterward.
    """
    part = f"subject_id={subject_id}"
    coalesced = f"{subject_id}.parquet"
    tmp = f"{subject_id}.parquet.tmp"

    fs, base = _partition_fs(output_prefix, part)
    entries = fs.list(base)
    session_files = [e for e in entries if e.endswith(".parquet") and e not in (coalesced, tmp)]
    if not session_files:
        return  # nothing new to merge (already coalesced)

    new_df = pd.concat([fs.read_parquet(e) for e in session_files], ignore_index=True)
    if coalesced in entries:
        old_df = fs.read_parquet(coalesced)
        old_df = old_df[~old_df["session_id"].isin(set(new_df["session_id"]))]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    use_cols = [c for c in sort_cols if c in combined.columns]
    if use_cols:
        combined = combined.sort_values(use_cols).reset_index(drop=True)

    fs.write_parquet(combined, tmp)
    for e in session_files:
        fs.delete(e)
    if coalesced in entries:
        fs.delete(coalesced)
    fs.rename(tmp, coalesced)


class _PartitionFS:
    """Minimal list/read/write/delete/rename over one partition dir (local or S3)."""

    def __init__(self, output_prefix, part):
        self.is_s3 = output_prefix.startswith("s3://")
        if self.is_s3:
            self._fs = s3fs.S3FileSystem(anon=False)
            self.base = f"{output_prefix[len('s3://'):]}/{part}"
        else:
            self.base = os.path.join(output_prefix, part)

    def list(self, _ignored=None):
        if self.is_s3:
            try:
                return [p.split("/")[-1] for p in self._fs.ls(self.base)]
            except FileNotFoundError:
                return []
        if not os.path.isdir(self.base):
            return []
        return os.listdir(self.base)

    def read_parquet(self, name):
        if self.is_s3:
            with self._fs.open(f"{self.base}/{name}", "rb") as f:
                return pd.read_parquet(f)
        return pd.read_parquet(os.path.join(self.base, name))

    def write_parquet(self, df, name):
        if self.is_s3:
            with self._fs.open(f"{self.base}/{name}", "wb") as f:
                df.to_parquet(f, index=False)
        else:
            df.to_parquet(os.path.join(self.base, name), index=False)

    def delete(self, name):
        if self.is_s3:
            self._fs.rm(f"{self.base}/{name}")
        else:
            os.remove(os.path.join(self.base, name))

    def rename(self, src, dst):
        if self.is_s3:
            self._fs.mv(f"{self.base}/{src}", f"{self.base}/{dst}")
        else:
            os.rename(os.path.join(self.base, src), os.path.join(self.base, dst))


def _partition_fs(output_prefix, part):
    """Return (_PartitionFS, base_path) for a subject partition dir."""
    fs = _PartitionFS(output_prefix, part)
    return fs, fs.base


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------


def _write_session_parquet(df, output_prefix, subject_id, session_id):  # noqa: C901
    """
    Write a single session's DataFrame to a Hive-partitioned parquet file.

    Output path: {output_prefix}/subject_id={subject_id}/{session_id}.parquet
    """
    df = df.copy()

    # Always store subject_id as string so it matches the Hive partition
    # directory name type and is consistent across all sources.
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)

    # Drop bpod_backup_* columns — raw hardware backup data not present in
    # the AIND reader output; keeping them causes schema conflicts and bloat.
    bpod_cols = [
        c for c in df.columns if any(c.startswith(pfx) for pfx in _TRIAL_COLS_TO_DROP_PREFIX)
    ]
    if bpod_cols:
        df = df.drop(columns=bpod_cols)

    # Apply canonical types to resolve cross-session type conflicts.
    # Determine which map to use: event tables always have 'event' column.
    canonical = _CANONICAL_EVENT_COL_TYPES if "event" in df.columns else _CANONICAL_TRIAL_COL_TYPES
    for col, target in canonical.items():
        if col not in df.columns:
            continue
        try:
            if target == "string":
                # Convert non-null values to str; store as pandas StringDtype
                # so that all-NaN float columns become string-null (not double)
                # which is what pa.Table.from_pandas stores as pa.string().
                df[col] = (
                    df[col].where(df[col].isna(), df[col].astype(str)).astype(pd.StringDtype())
                )
            elif target == "double":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif target == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        except Exception as exc:
            logger.warning("Type normalisation failed for column %r: %s", col, exc)

    # Sanitise object columns:
    #   1. Convert array-like values to their string representation so PyArrow
    #      can store them (lick-time lists etc.).
    #   2. Cast mixed string+number columns to string (e.g. event ``data``
    #      column that holds 'earned'/'auto' alongside numeric values).
    #   3. Drop all-null object columns — pynwb sometimes returns these as
    #      float NaN, which PyArrow stores as ``double``.  Across sessions the
    #      same column may be ``string`` in one file and ``double`` in another,
    #      which breaks cross-file schema merging.  Omitting the column
    #      entirely is safe: PyArrow fills it with ``null`` when merging.
    cols_to_drop = []
    for col in df.select_dtypes(include="object").columns:
        sample_vals = df[col].dropna()
        if len(sample_vals) == 0:
            cols_to_drop.append(col)
            continue
        first = sample_vals.iloc[0]
        if hasattr(first, "__len__") and not isinstance(first, str):
            df[col] = df[col].apply(
                lambda x: str(list(x)) if hasattr(x, "__len__") and not isinstance(x, str) else x
            )
        else:
            has_str = sample_vals.apply(type).eq(str).any()
            has_num = sample_vals.apply(lambda v: isinstance(v, (int, float))).any()
            if has_str and has_num:
                df[col] = df[col].astype(str)
    df = df.drop(columns=cols_to_drop)

    table = pa.Table.from_pandas(df, preserve_index=False)
    partition_dir = f"subject_id={subject_id}"

    if output_prefix.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        s3_base = output_prefix[len("s3://") :]
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
        s3_path = path[len("s3://") :]
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
        with fs.open(path[len("s3://") :], "w") as f:
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
        with fs.open(path[len("s3://") :], "r") as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)


def _read_csv(path):
    """Read a CSV from a local path or S3 URI. Raises FileNotFoundError if absent."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        s3_path = path[len("s3://") :]
        if not fs.exists(s3_path):
            raise FileNotFoundError(path)
        with fs.open(s3_path, "r") as f:
            return pd.read_csv(f)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(df, path):
    """Write a DataFrame as CSV to a local path or S3 URI."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path[len("s3://") :], "w") as f:
            df.to_csv(f, index=False)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(path, index=False)
