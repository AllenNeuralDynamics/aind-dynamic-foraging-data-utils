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
"""

from aind_dynamic_foraging_data_utils.foraging_cache.util.parquet_builder import (  # noqa: F401
    build_nwb_file_index,
    build_session_table,
    build_trial_and_event_tables,
)
