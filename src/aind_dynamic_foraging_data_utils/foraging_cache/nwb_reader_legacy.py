"""
Legacy NWB reader — adapted from Han pipeline ``process_nwbs.py``.

This reader handles both bonsai and bpod NWB files using a simpler trial-
extraction approach that avoids the assertion-heavy AIND reader.

Key differences from the AIND reader (``nwb_utils.create_df_trials``):
  - Does NOT access ``bpod_backup_BehavioralEvents`` (which is malformed in
    many old bpod NWBs and crashes pynwb).
  - Skips the reward-before-choice-time and reward-time sanity checks that
    fail on 2025+ bonsai NWBs.
  - Falls back to raw h5py reading if pynwb cannot load the file at all.

The trade-off is that the output schema is simpler: lick times are stored as
arrays per trial (``left_lick_time``, ``right_lick_time``), and reward-type
annotations may be absent for older sessions.

Source:
    https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/blob/main/code/process_nwbs.py

"""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_nwb_pynwb(nwb_path):
    """Load an NWB file with pynwb (HDF5) or hdmf_zarr (Zarr/S3)."""
    import os

    from hdmf_zarr import NWBZarrIO
    from pynwb import NWBHDF5IO

    if os.path.isdir(nwb_path) or (nwb_path.startswith("s3://") and nwb_path.endswith(".nwb")):
        io = NWBZarrIO(nwb_path, mode="r")
    else:
        io = NWBHDF5IO(nwb_path, mode="r")
    return io.read(), io


def _load_trials_h5py(nwb_path):
    """
    Fallback: read the /intervals/trials group directly with h5py.

    This bypasses pynwb entirely and avoids crashes on files with malformed
    TimeSeries objects (e.g. bpod_backup_BehavioralEvents missing 'timestamps').

    Returns a DataFrame with the same columns as nwb.trials.to_dataframe().
    """
    import h5py

    with h5py.File(nwb_path, "r") as f:
        trials_grp = f["intervals/trials"]
        col_names = [k for k in trials_grp.keys() if k != "id"]
        data = {}
        n_rows = None
        for col in col_names:
            dset = trials_grp[col]
            vals = dset[:]
            # h5py returns bytes for string datasets
            if vals.dtype.kind == "S" or vals.dtype.kind == "O":
                vals = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in vals])
            data[col] = vals
            if n_rows is None:
                n_rows = len(vals)

        if "id" in trials_grp:
            ids = trials_grp["id"][:]
        else:
            ids = np.arange(n_rows)

        df = pd.DataFrame(data, index=ids)
        df.index.name = "id"
    return df


def _load_acquisition_timestamps_h5py(nwb_path, key):
    """Read timestamps from /acquisition/{key}/timestamps using h5py."""
    import h5py

    with h5py.File(nwb_path, "r") as f:
        grp = f[f"acquisition/{key}"]
        if "timestamps" in grp:
            return grp["timestamps"][:]
        return np.array([])


def read_trials(nwb_path):
    """
    Read trial table from an NWB file using the legacy Han pipeline approach.

    Tries pynwb first; falls back to h5py if pynwb cannot construct the file
    (common with old bpod NWBs that have malformed TimeSeries objects).

    The output DataFrame contains columns from ``nwb.trials`` plus computed
    columns: ``reward``, ``reward_non_autowater``, ``non_autowater_trial``.

    Args:
        nwb_path (str): Path to an NWB (HDF5) file.

    Returns:
        pd.DataFrame: Trial table.
    """
    nwb = None
    io = None
    use_h5py_fallback = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nwb, io = _load_nwb_pynwb(nwb_path)
    except Exception as exc:
        logger.info("pynwb failed for %s (%s), falling back to h5py", nwb_path, exc)
        use_h5py_fallback = True

    try:
        if use_h5py_fallback:
            return _compute_df_trial_h5py(nwb_path)
        else:
            return _compute_df_trial(nwb, nwb_path)
    finally:
        if io is not None:
            try:
                io.close()
            except Exception:
                pass


def read_events(nwb_path):
    """
    Read event table from an NWB file using the legacy Han pipeline approach.

    Tries pynwb first; falls back to h5py for old bpod files.

    Args:
        nwb_path (str): Path to an NWB (HDF5) file.

    Returns:
        pd.DataFrame: Tidy event table with columns [timestamps, data, event].
    """
    nwb = None
    io = None
    use_h5py_fallback = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nwb, io = _load_nwb_pynwb(nwb_path)
    except Exception as exc:
        logger.info("pynwb failed for %s (%s), falling back to h5py", nwb_path, exc)
        use_h5py_fallback = True

    try:
        if use_h5py_fallback:
            return _compute_df_events_h5py(nwb_path)
        else:
            return _compute_df_events(nwb)
    finally:
        if io is not None:
            try:
                io.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Internal: pynwb-based extraction (adapted from process_nwbs.py)
# ---------------------------------------------------------------------------


def _compute_df_trial(nwb, nwb_path):
    """
    Extract trial table from a loaded NWB object.

    Adapted from Han's ``compute_df_trial()`` in process_nwbs.py.
    Does NOT access ``bpod_backup_BehavioralEvents``.
    """
    df_trial = nwb.trials.to_dataframe().copy()

    # --- Reward columns ---
    if "earned_reward" not in df_trial.columns:
        if "rewarded_historyL" in df_trial.columns and "rewarded_historyR" in df_trial.columns:
            df_trial["earned_reward"] = (
                df_trial["rewarded_historyL"] | df_trial["rewarded_historyR"]
            )
        else:
            df_trial["earned_reward"] = False

    # Compute autowater vs earned distinction
    if "auto_waterL" in df_trial.columns and "auto_waterR" in df_trial.columns:
        df_trial["non_autowater_trial"] = ~(df_trial["auto_waterL"] | df_trial["auto_waterR"])
        df_trial["reward_non_autowater"] = (
            df_trial["earned_reward"] & df_trial["non_autowater_trial"]
        )
    else:
        df_trial["non_autowater_trial"] = True
        df_trial["reward_non_autowater"] = df_trial["earned_reward"]

    # --- Lick times from nwb.acquisition ---
    lick_keys = ["left_lick_time", "right_lick_time"]
    for key in lick_keys:
        if key in nwb.acquisition:
            timestamps = nwb.acquisition[key].timestamps[:]
            # Map lick timestamps to trials using goCue_start_time boundaries
            if "goCue_start_time" in df_trial.columns:
                go_cues = df_trial["goCue_start_time"].values
                lick_lists = _map_events_to_trials(timestamps, go_cues)
                df_trial[key] = lick_lists
            else:
                df_trial[key] = [np.array([])] * len(df_trial)
        else:
            df_trial[key] = [np.array([])] * len(df_trial)

    # --- Reaction time ---
    if "goCue_start_time" in df_trial.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_trial["reaction_time"] = df_trial.apply(
                lambda row: (
                    _first_lick_time(row) - row["goCue_start_time"]
                    if _first_lick_time(row) is not np.nan
                    else np.nan
                ),
                axis=1,
            )

    return df_trial


def _compute_df_events(nwb):
    """
    Extract event table from a loaded NWB object.

    Reads all acquisition TimeSeries that have timestamps (skipping FIP
    channels and sniff_detector), similar to ``create_df_events()`` in
    nwb_utils but without the assertion checks.
    """
    event_types = set(nwb.acquisition.keys())

    # Filter FIP channels
    channels = ["G", "R", "Iso"]
    fibers = ["0", "1", "2", "3", "4"]
    fip_prefixes = [f"{c}_{f}" for c in channels for f in fibers]
    event_types = {k for k in event_types if not any(k.startswith(pfx) for pfx in fip_prefixes)}
    event_types -= {"FIP_falling_time", "FIP_rising_time", "sniff_detector"}

    events = []
    for e in event_types:
        try:
            raw_stamps = nwb.acquisition[e].timestamps[:]
            data = nwb.acquisition[e].data[:]
            df = pd.DataFrame(
                {
                    "timestamps": raw_stamps,
                    "data": data,
                    "event": e,
                }
            )
            events.append(df)
        except Exception:
            continue

    if not events:
        return pd.DataFrame(columns=["timestamps", "data", "event"])

    df = pd.concat(events, ignore_index=True)
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Internal: h5py fallback for files pynwb cannot load
# ---------------------------------------------------------------------------


def _compute_df_trial_h5py(nwb_path):
    """
    Extract trial table using raw h5py (bypasses pynwb entirely).

    Produces a simpler output: columns from /intervals/trials plus
    earned_reward computed from rewarded_historyL/R if available.
    """
    df_trial = _load_trials_h5py(nwb_path)

    # Compute earned_reward
    if "rewarded_historyL" in df_trial.columns and "rewarded_historyR" in df_trial.columns:
        df_trial["earned_reward"] = df_trial["rewarded_historyL"].astype(bool) | df_trial[
            "rewarded_historyR"
        ].astype(bool)
    else:
        df_trial["earned_reward"] = False

    # Lick times from acquisition
    for key in ["left_lick_time", "right_lick_time"]:
        timestamps = _load_acquisition_timestamps_h5py(nwb_path, key)
        if len(timestamps) > 0 and "goCue_start_time" in df_trial.columns:
            go_cues = df_trial["goCue_start_time"].values.astype(float)
            df_trial[key] = _map_events_to_trials(timestamps, go_cues)
        else:
            df_trial[key] = [np.array([])] * len(df_trial)

    return df_trial


def _compute_df_events_h5py(nwb_path):
    """
    Extract event table using raw h5py.

    Reads all groups under /acquisition/ that have a 'timestamps' dataset.
    """
    import h5py

    events = []
    with h5py.File(nwb_path, "r") as f:
        if "acquisition" not in f:
            return pd.DataFrame(columns=["timestamps", "data", "event"])

        for key in f["acquisition"].keys():
            # Skip FIP channels
            if any(
                key.startswith(pfx)
                for pfx in [
                    f"{c}_{fi}" for c in ["G", "R", "Iso"] for fi in ["0", "1", "2", "3", "4"]
                ]
            ):
                continue
            if key in ("FIP_falling_time", "FIP_rising_time", "sniff_detector"):
                continue

            grp = f[f"acquisition/{key}"]
            if "timestamps" not in grp:
                continue
            try:
                stamps = grp["timestamps"][:]
                data = grp["data"][:] if "data" in grp else np.ones(len(stamps))
                df = pd.DataFrame(
                    {
                        "timestamps": stamps,
                        "data": data,
                        "event": key,
                    }
                )
                events.append(df)
            except Exception:
                continue

    if not events:
        return pd.DataFrame(columns=["timestamps", "data", "event"])

    df = pd.concat(events, ignore_index=True)
    return df.sort_values("timestamps").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _map_events_to_trials(event_times, go_cue_times):
    """
    Map event timestamps to trials based on goCue boundaries.

    Returns a list of arrays, one per trial, containing the event timestamps
    that fall within [goCue_i, goCue_{i+1}).
    """
    n_trials = len(go_cue_times)
    result = []
    for i in range(n_trials):
        start = go_cue_times[i]
        end = go_cue_times[i + 1] if i + 1 < n_trials else np.inf
        mask = (event_times >= start) & (event_times < end)
        result.append(event_times[mask])
    return result


def _first_lick_time(row):
    """Return the earliest lick time across left and right lick arrays."""
    times = []
    for key in ["left_lick_time", "right_lick_time"]:
        arr = row.get(key, np.array([]))
        if hasattr(arr, "__len__") and len(arr) > 0:
            times.append(arr[0])
    return min(times) if times else np.nan
