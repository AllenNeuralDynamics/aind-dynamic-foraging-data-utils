"""
AIND pipeline NWB reader — thin wrapper around nwb_utils.

Calls ``create_df_trials()`` and ``create_df_events()`` from
:mod:`aind_dynamic_foraging_data_utils.nwb_utils`, but converts the
AssertionError / AssertionError exceptions those functions raise on data-
quality issues into a single :class:`AINDReaderQualityError` so the caller
can fall back to the legacy reader without catching broad ``Exception``.

Known assertion-error messages from ``create_df_trials()``:
  - "Rewarded trials without reward time"
  - "Reward before choice time"
  - "Unrewarded trials with reward time"

These typically appear on post-2025-01-01 bonsai NWBs where the new AIND
pipeline introduced regressions.
"""

import warnings


class AINDReaderQualityError(Exception):
    """Raised when the AIND pipeline reader detects a data-quality problem."""


def read_trials(nwb_path, **kwargs):
    """
    Read trial table from an NWB file using the AIND pipeline reader.

    Wraps ``nwb_utils.create_df_trials(nwb_path, verbose=False, **kwargs)``.
    Any ``AssertionError`` raised by the AIND reader is re-raised as
    :class:`AINDReaderQualityError` so the caller can catch it cleanly.

    Args:
        nwb_path (str): Path or S3 URI of an NWB file.
        **kwargs: Forwarded to ``create_df_trials()``.

    Returns:
        pd.DataFrame: Trial table in AIND minimal schema.

    Raises:
        AINDReaderQualityError: On assertion failures from data-quality checks.
    """
    from aind_dynamic_foraging_data_utils import nwb_utils

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nwb_utils.create_df_trials(nwb_path, verbose=False, **kwargs)
    except AssertionError as exc:
        raise AINDReaderQualityError(str(exc)) from exc


def read_events(nwb_path, **kwargs):
    """
    Read event table from an NWB file using the AIND pipeline reader.

    Wraps ``nwb_utils.create_df_events(nwb_path, verbose=False, **kwargs)``.

    Args:
        nwb_path (str): Path or S3 URI of an NWB file.
        **kwargs: Forwarded to ``create_df_events()``.

    Returns:
        pd.DataFrame: Tidy event table.

    Raises:
        AINDReaderQualityError: On assertion failures from data-quality checks.
    """
    from aind_dynamic_foraging_data_utils import nwb_utils

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nwb_utils.create_df_events(nwb_path, verbose=False, **kwargs)
    except AssertionError as exc:
        raise AINDReaderQualityError(str(exc)) from exc
