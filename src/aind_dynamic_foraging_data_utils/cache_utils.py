"""
Query API for the foraging parquet cache on S3.

Provides fast, filter-pushdown queries against pre-built parquet tables.
The trial and event tables use Hive partitioning (subject_id=<id>) so that
reading data for specific mice never touches irrelevant partitions.

S3 source: s3://aind-behavior-data/foraging_cache/

Tables:
  session_table.parquet         — one row per session (all session-level metadata)
  trial_table/subject_id=<id>/  — one row per trial (AIND minimal schema)
  event_table/subject_id=<id>/  — one row per behavioral event

Usage:
    from aind_dynamic_foraging_data_utils import cache_utils

    # All sessions for two mice
    df_sess = cache_utils.get_session_table(subject_ids=["633456", "712345"])

    # Trial data for those same mice — reads only their partitions
    df_trials = cache_utils.get_trial_table(subject_ids=["633456", "712345"])

    # Filter by date range
    df_trials = cache_utils.get_trial_table(
        subject_ids=["633456"],
        date_range=("2024-01-01", "2024-12-31"),
    )
"""

from typing import List, Optional, Tuple

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs

# ---- Default S3 paths ----
_S3_BUCKET = "aind-behavior-data"
_S3_PREFIX = "foraging_cache"
DEFAULT_SESSION_TABLE_URI = f"s3://{_S3_BUCKET}/{_S3_PREFIX}/session_table.parquet"
DEFAULT_TRIAL_TABLE_PREFIX = f"s3://{_S3_BUCKET}/{_S3_PREFIX}/trial_table"
DEFAULT_EVENT_TABLE_PREFIX = f"s3://{_S3_BUCKET}/{_S3_PREFIX}/event_table"


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def get_session_table(
    subject_ids: Optional[List[str]] = None,
    date_range: Optional[Tuple[str, str]] = None,
    exclude_bad_bowen: bool = True,
    session_table_uri: str = DEFAULT_SESSION_TABLE_URI,
) -> pd.DataFrame:
    """
    Load the session-level metadata table from the foraging cache.

    Uses pyarrow predicate pushdown on subject_id and is_bad_bowen_session where
    possible; date filtering is applied in-memory after loading.

    Args:
        subject_ids (list[str] | None): Subject IDs to include. None returns all subjects.
            Pass strings (e.g. ["633456"]) — ints are auto-converted.
        date_range (tuple | None): (start_date, end_date) as "YYYY-MM-DD" strings,
            inclusive on both ends. None returns all dates.
        exclude_bad_bowen (bool): If True (default), exclude sessions flagged
            is_bad_bowen_session=True (Bowen sessions with unreliable bonsai data).
        session_table_uri (str): S3 URI or local path of the session parquet file.

    Returns:
        pd.DataFrame sorted by (subject_id, session_date, nwb_suffix). One row per session.

    Example:
        df = get_session_table(
            subject_ids=["633456"],
            date_range=("2024-01-01", "2024-12-31"),
        )
    """
    # Build pyarrow row-group filter
    filters = []
    if subject_ids is not None:
        filters.append(("subject_id", "in", [str(s) for s in subject_ids]))
    if exclude_bad_bowen:
        filters.append(("is_bad_bowen_session", "==", False))

    fs, path = _parse_uri(session_table_uri)

    if session_table_uri.startswith("s3://"):
        with fs.open(path, "rb") as f:
            table = pq.read_table(f, filters=filters if filters else None)
    else:
        table = pq.read_table(path, filters=filters if filters else None)

    df = table.to_pandas()

    # Date range is applied in-memory (string comparison works for ISO dates)
    if date_range is not None:
        start, end = date_range
        df = df[(df["session_date"] >= start) & (df["session_date"] <= end)]

    return df.sort_values(["subject_id", "session_date", "nwb_suffix"]).reset_index(drop=True)


def get_trial_table(
    subject_ids: List[str],
    date_range: Optional[Tuple[str, str]] = None,
    trial_table_prefix: str = DEFAULT_TRIAL_TABLE_PREFIX,
) -> pd.DataFrame:
    """
    Load trial-level data from the foraging cache for the specified subjects.

    Reads only the Hive partitions for the requested subject_ids, so this is
    fast even for very large total datasets.

    Args:
        subject_ids (list[str]): Subject IDs (required). Used for Hive partition pruning.
            Pass strings or ints — ints are auto-converted.
        date_range (tuple | None): Optional (start_date, end_date) "YYYY-MM-DD" filter.
        trial_table_prefix (str): S3 URI or local directory of the Hive-partitioned parquet.

    Returns:
        pd.DataFrame sorted by (subject_id, session_date, nwb_suffix, trial).
        One row per trial with AIND minimal schema columns.

    Example:
        df = get_trial_table(subject_ids=["633456", "712345"])
    """
    df = _read_hive_partitioned(trial_table_prefix, subject_ids, date_range)
    sort_cols = [c for c in ["subject_id", "session_date", "nwb_suffix", "trial"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def get_event_table(
    subject_ids: List[str],
    date_range: Optional[Tuple[str, str]] = None,
    event_table_prefix: str = DEFAULT_EVENT_TABLE_PREFIX,
) -> pd.DataFrame:
    """
    Load event-level data from the foraging cache for the specified subjects.

    Reads only the Hive partitions for the requested subject_ids.

    Args:
        subject_ids (list[str]): Subject IDs (required).
        date_range (tuple | None): Optional (start_date, end_date) "YYYY-MM-DD" filter.
        event_table_prefix (str): S3 URI or local directory of the Hive-partitioned parquet.

    Returns:
        pd.DataFrame sorted by (subject_id, session_id, timestamps).
        One row per behavioral event.

    Example:
        df = get_event_table(subject_ids=["633456"])
    """
    df = _read_hive_partitioned(event_table_prefix, subject_ids, date_range)
    sort_cols = [c for c in ["subject_id", "session_id", "timestamps"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_hive_partitioned(
    prefix: str,
    subject_ids: List[str],
    date_range: Optional[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Read a Hive-partitioned parquet dataset with filter pushdown.

    Constructs a PyArrow Dataset pointing at {prefix}/subject_id={id}/ directories.
    The subject_id filter is pushed down to the partition level, so S3 I/O is
    proportional to the number of requested subjects, not the total dataset size.

    Args:
        prefix (str): S3 URI or local directory with Hive-partitioned parquet.
        subject_ids (list): Subject IDs (strings or ints).
        date_range (tuple | None): Optional (start_date, end_date) strings.

    Returns:
        pd.DataFrame with all requested partitions concatenated.
    """
    subject_ids_str = [str(s) for s in subject_ids]
    fs, base_path = _parse_uri(prefix)

    # Build PyArrow filter expression combining subject_id partition filter
    # and optional session_date range filter (applied as a row-group predicate).
    filter_expr = ds.field("subject_id").isin(subject_ids_str)

    if date_range is not None:
        start, end = date_range
        filter_expr = (
            filter_expr
            & (ds.field("session_date") >= start)
            & (ds.field("session_date") <= end)
        )

    # Explicitly declare partition schema as string to avoid PyArrow auto-inferring
    # subject_id as int32 from directory names that look numeric (e.g. "subject_id=697062").
    import pyarrow as pa

    partition_schema = ds.partitioning(
        pa.schema([("subject_id", pa.string())]), flavor="hive"
    )

    dataset = ds.dataset(
        base_path,
        filesystem=fs,
        format="parquet",
        partitioning=partition_schema,
    )

    table = dataset.to_table(filter=filter_expr)
    return table.to_pandas()


def _parse_uri(uri: str):
    """
    Parse a URI and return (filesystem, path) usable by PyArrow.

    For S3 URIs (s3://...) returns an s3fs filesystem and the bare path.
    For local paths returns the PyArrow local filesystem and the path unchanged.
    """
    if uri.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        return fs, uri[len("s3://"):]
    else:
        import pyarrow.fs as pafs

        return pafs.LocalFileSystem(), uri
