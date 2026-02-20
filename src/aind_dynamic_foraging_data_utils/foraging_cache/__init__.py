"""
Foraging parquet cache — batch builder and legacy NWB readers.

This sub-package isolates the batch database builder from the rest of the
library.  It contains:

  parquet_builder     – builds session / trial / event parquet tables
  nwb_reader_aind     – thin wrapper around nwb_utils (AIND pipeline reader)
  nwb_reader_legacy   – Han pipeline reader adapted from process_nwbs.py
"""

from aind_dynamic_foraging_data_utils.foraging_cache.parquet_builder import (  # noqa: F401
    build_nwb_file_index,
    build_session_table,
    build_trial_and_event_tables,
)
