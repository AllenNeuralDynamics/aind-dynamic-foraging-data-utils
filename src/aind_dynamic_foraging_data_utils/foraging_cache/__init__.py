"""
Foraging parquet cache.

Entry points (top level):
  build_cache      – build / incrementally extend the parquet database
  query_examples   – read-back ("return loop") DuckDB query patterns

Implementation modules live in the ``util`` sub-package:
  util.parquet_builder   – builds session / trial / event parquet tables
  util.nwb_reader_aind   – thin wrapper around nwb_utils (AIND pipeline reader)
  util.nwb_reader_legacy – Han pipeline reader adapted from process_nwbs.py
  util.postprocess       – per-subject coalescing + triage CSV log

Default read targets for the production database (``SESSION_DB`` / ``TRIAL_DB`` /
``EVENT_DB``) live on a public S3 bucket, so reading needs no AWS credentials::

    import duckdb
    from aind_dynamic_foraging_data_utils.foraging_cache import SESSION_DB, TRIAL_DB
    duckdb.sql(f"SELECT * FROM read_parquet('{SESSION_DB}')").df()
"""

from aind_dynamic_foraging_data_utils.foraging_cache.util.parquet_builder import (  # noqa: F401
    build_nwb_file_index,
    build_session_table,
    build_trial_and_event_tables,
)

# Canonical production parquet database on S3 (public bucket — no credentials to read).
# Build/extend it with ``build_cache``; reassign these to query a local build instead.
PROD_S3_PREFIX = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
SESSION_DB = f"{PROD_S3_PREFIX}/session_table.parquet"  # flat session table
TRIAL_DB = f"{PROD_S3_PREFIX}/trial_table"  # Hive-partitioned by subject_id
EVENT_DB = f"{PROD_S3_PREFIX}/event_table"  # Hive-partitioned by subject_id
