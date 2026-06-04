"""
Post-processing for the foraging parquet cache: per-subject coalescing and the
human-readable triage log.

Split out of parquet_builder so the *build* step (NWB -> per-session parquet) is
separate from the *output-shaping / logging* step:
  - coalesce_partitions : merge each subject's per-session files into one sorted
    file (single-session queries stay one S3 GET; full-load file count drops ~26x).
  - append_triage_log   : maintain a cumulative human-readable per-session CSV.
  - read_csv / write_csv: small local-or-S3 CSV I/O helpers used by both.
"""

import datetime as _dt
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)

# Columns of the human-readable per-session triage log (one row per session).
TRIAGE_COLUMNS = [
    "session_id", "subject_id", "session_date", "nwb_suffix",
    "status", "data_source", "reader", "n_trials", "n_events",
    "nwb_path", "error", "processed_at",
]


# ---------------------------------------------------------------------------
# Triage log
# ---------------------------------------------------------------------------


def append_triage_log(log_csv_path, results, verbose=True):
    """
    Write/merge a human-readable CSV with one row per processed session.

    Each row records the triage detail (status, data source, reader used, row
    counts, NWB path, error). Existing rows are merged in and de-duplicated by
    session_id (the latest run wins), so the file is a cumulative state of every
    session ever processed.
    """
    now = _dt.datetime.now().isoformat(timespec="seconds")
    new_rows = [{**r, "processed_at": now} for r in results]
    df_new = pd.DataFrame(new_rows).reindex(columns=TRIAGE_COLUMNS)

    # Merge with any existing log (latest row per session_id wins).
    try:
        df_old = read_csv(log_csv_path)
        df_old = df_old[~df_old["session_id"].isin(set(df_new["session_id"]))]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all = df_all.sort_values(["subject_id", "session_date", "nwb_suffix"]).reset_index(drop=True)
    write_csv(df_all, log_csv_path)
    if verbose:
        print(f"  Triage log: {len(df_new)} updated -> {log_csv_path} ({len(df_all)} total)")


# ---------------------------------------------------------------------------
# Per-subject coalescing
# ---------------------------------------------------------------------------


def coalesce_partitions(output_prefix, subject_ids, sort_cols, n_threads=16, verbose=True):
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

    fs = _PartitionFS(output_prefix, part)
    entries = fs.list()
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
        """Bind to one partition dir under ``output_prefix`` (S3 or local)."""
        self.is_s3 = output_prefix.startswith("s3://")
        if self.is_s3:
            self._fs = s3fs.S3FileSystem(anon=False)
            self.base = f"{output_prefix[len('s3://'):]}/{part}"
        else:
            self.base = os.path.join(output_prefix, part)

    def list(self):
        """Return the file names in the partition dir (empty if it doesn't exist)."""
        if self.is_s3:
            try:
                return [p.split("/")[-1] for p in self._fs.ls(self.base)]
            except FileNotFoundError:
                return []
        if not os.path.isdir(self.base):
            return []
        return os.listdir(self.base)

    def read_parquet(self, name):
        """Read one parquet file from the partition into a DataFrame."""
        if self.is_s3:
            with self._fs.open(f"{self.base}/{name}", "rb") as f:
                return pd.read_parquet(f)
        return pd.read_parquet(os.path.join(self.base, name))

    def write_parquet(self, df, name):
        """Write a DataFrame to one parquet file in the partition."""
        if self.is_s3:
            with self._fs.open(f"{self.base}/{name}", "wb") as f:
                df.to_parquet(f, index=False)
        else:
            df.to_parquet(os.path.join(self.base, name), index=False)

    def delete(self, name):
        """Delete one file from the partition."""
        if self.is_s3:
            self._fs.rm(f"{self.base}/{name}")
        else:
            os.remove(os.path.join(self.base, name))

    def rename(self, src, dst):
        """Rename a file within the partition."""
        if self.is_s3:
            self._fs.mv(f"{self.base}/{src}", f"{self.base}/{dst}")
        else:
            os.rename(os.path.join(self.base, src), os.path.join(self.base, dst))


# ---------------------------------------------------------------------------
# CSV I/O (local or S3)
# ---------------------------------------------------------------------------


def read_csv(path):
    """Read a CSV from a local path or S3 URI. Raises FileNotFoundError if absent."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        s3_path = path[len("s3://"):]
        if not fs.exists(s3_path):
            raise FileNotFoundError(path)
        with fs.open(s3_path, "r") as f:
            return pd.read_csv(f)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv(df, path):
    """Write a DataFrame as CSV to a local path or S3 URI."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path[len("s3://"):], "w") as f:
            df.to_csv(f, index=False)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(path, index=False)
