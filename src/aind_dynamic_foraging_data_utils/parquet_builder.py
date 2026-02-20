"""
Backwards-compatibility shim — all builder code has moved to
:mod:`aind_dynamic_foraging_data_utils.foraging_cache.parquet_builder`.

Import from there directly for new code::

    from aind_dynamic_foraging_data_utils.foraging_cache import parquet_builder
"""

from aind_dynamic_foraging_data_utils.foraging_cache.parquet_builder import (  # noqa: F401
    BUILD_METADATA_S3_URI,
    BOWEN_INCOMPLETE_CSV,
    EVENT_TABLE_S3_PREFIX,
    LOCAL_BONSAI_NWB_DIR,
    LOCAL_BPOD_NWB_DIR,
    S3_CACHE_BUCKET,
    S3_CACHE_PREFIX,
    SESSION_TABLE_S3_URI,
    TRIAL_TABLE_S3_PREFIX,
    _parse_nwb_filename,
    build_nwb_file_index,
    build_session_table,
    build_trial_and_event_tables,
)
