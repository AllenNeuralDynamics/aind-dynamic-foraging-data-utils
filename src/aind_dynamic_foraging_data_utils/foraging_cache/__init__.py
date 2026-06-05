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

Reading uses the query helpers in ``query`` (DuckDB under the hood). Reach for the simple
helpers first; drop to native SQL when you need more::

    from aind_dynamic_foraging_data_utils.foraging_cache import select_sessions, fetch_trials
    sel = select_sessions("task LIKE '%Uncoupled%' AND foraging_eff > 0.8")
    trials = fetch_trials(sel, columns=["animal_response", "earned_reward"])

The default read targets (``SESSION_DB`` / ``TRIAL_DB`` / ``EVENT_DB``) live on a public S3
bucket, so reading needs no AWS credentials.
"""

from aind_dynamic_foraging_data_utils.foraging_cache.util.parquet_builder import (  # noqa: F401
    build_nwb_file_index,
    build_session_table,
    build_trial_and_event_tables,
)

# Read API (DuckDB query helpers) + the canonical production DB paths on S3 (public
# bucket — no credentials to read). Build/extend the DB with ``build_cache``; pass
# ``base=`` to the helpers (or reassign these) to query a local build instead.
from aind_dynamic_foraging_data_utils.foraging_cache.query import (  # noqa: F401
    EVENT_DB,
    PROD_S3_PREFIX,
    SESSION_DB,
    TRIAL_DB,
    fetch_events,
    fetch_trials,
    read_events,
    read_trials,
    select_sessions,
)
