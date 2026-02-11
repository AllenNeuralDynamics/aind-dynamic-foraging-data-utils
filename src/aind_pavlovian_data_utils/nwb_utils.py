"""
Utility functions for processing dynamic foraging data.
    load_nwb_from_filename
    create_single_df_session_inner
    create_df_session
    create_single_df_session
    create_df_trials
    create_df_events
    create_df_fip
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from datetime import date

# If we adjust time_in_session, adjust it to this
SESSION_ALIGNMENT = "CS_start_time"

# ms to s conversion 
MS_TO_S = 1000


def load_nwb_from_filename(filename):
    """
    Load NWB from file, checking for HDF5 or Zarr
    if filename is not a string, then return the input, assuming its the NWB file already
    """

    if type(filename) is str:
        if os.path.isdir(filename) or (filename.startswith("s3://") and filename.endswith(".nwb")):
            io = NWBZarrIO(filename, mode="r")
            nwb = io.read()
            return nwb
        elif os.path.isfile(filename):
            io = NWBHDF5IO(filename, mode="r")
            nwb = io.read()
            return nwb
        else:
            raise FileNotFoundError(filename)
    else:
        # Assuming its already an NWB
        return filename


def create_single_df_session_inner(nwb):
    """
    given a nwb file, output a tidy dataframe
    """
    df_trials = nwb.trials.to_dataframe()

    # -- Session-based table --
    # - Meta data -
    session_start_time_from_meta = nwb.session_start_time
    session_date_from_meta = session_start_time_from_meta.strftime("%Y-%m-%d")
    subject_id_from_meta = nwb.subject.subject_id

    # TODO, should reprocess old files, and remove this logic
    if "behavior" in nwb.session_id:
        splits = nwb.session_id.split("_")
        subject_id = splits[1]
        session_date = splits[2]
        nwb_suffix = splits[3].replace("-", "")
    else:
        old_re = re.match(
            r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<n>\d+))?\.json",
            nwb.session_id,
        )

        if old_re is not None:
            # If there are more than one "bonsai sessions" (the trainer clicked "Save" button in the
            # GUI more than once) in a certain day,
            # parse nwb_suffix from the file name (0, 1, 2, ...)
            subject_id, session_date, nwb_suffix = old_re.groups()
            nwb_suffix = int(nwb_suffix) if nwb_suffix is not None else 0
        else:
            # After https://github.com/AllenNeuralDynamics/dynamic-foraging-task/commit/
            # 62d0e9e2bb9b47a8efe8ecb91da9653381a5f551,
            # the suffix becomes the session start time. Therefore, I use HHMMSS as the nwb suffix,
            # which still keeps the order as before.

            # Typical situation for multiple bonsai sessions per day is that the
            # RAs pressed more than once "Save" button but only started the session once.
            # Therefore, I should generate nwb_suffix from the bonsai file name
            # instead of session_start_time.
            subject_id, session_date, session_json_time = re.match(
                r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>.*))\.json",
                nwb.session_id,
            ).groups()
            nwb_suffix = int(session_json_time.replace("-", ""))

    # Ad-hoc bug fixes for some mistyped mouse ID

    assert subject_id == subject_id_from_meta, (
        f"Subject name from the metadata ({subject_id_from_meta}) does not match "
        f"that from json name ({subject_id})!!"
    )
    assert session_date == session_date_from_meta, (
        f"Session date from the metadata ({session_date_from_meta}) does not match "
        f"that from json name ({session_date})!!"
    )

    session_index = pd.MultiIndex.from_tuples(
        [(subject_id, session_date, nwb_suffix)],
        names=["subject_id", "session_date", "nwb_suffix"],
    )

    # Parse meta info
    # TODO: when generating nwb, put meta info in nwb.scratch and get rid of the regular expression
    # This could be significantly cleaned up based on new metadata format
    # But im making it consistent for now
    # TODO, should reprocess old files, and remove this logic
    if "behavior" not in nwb.session_id:
        extra_water, rig = re.search(
            r"Give extra water.*:(\d*(?:\.\d+)?)? .*?(?:tower|box):(.*)?",
            nwb.session_description,
        ).groups()
        weight_after_session = re.search(
            r"Weight after.*:(\d*(?:\.\d+)?)?", nwb.subject.description
        ).groups()[0]

        extra_water = float(extra_water) if extra_water != "" else 0
        weight_after_session = float(weight_after_session) if weight_after_session != "" else np.nan
        weight_before_session = float(nwb.subject.weight) if nwb.subject.weight != "" else np.nan
        user_name = nwb.experimenter[0]
    elif nwb.scratch:
        rig = nwb.scratch["metadata"].box[0]
        user_name = nwb.experimenter
        weight_after_session = nwb.scratch["metadata"].weight_after[0]
        water_during_session = nwb.scratch["metadata"].water_in_session_total[0]
        weight_before_session = weight_after_session - water_during_session
        extra_water = nwb.scratch["metadata"].water_in_session_manual[0]
    else:
        rig = 0
        user_name = 0
        weight_after_session = 0
        water_during_session = 0
        weight_before_session = 0
        extra_water = 0

    dict_meta = {
        "rig": rig,
        "user_name": user_name,
        "experiment_description": nwb.experiment_description,
        "task": nwb.protocol,
        "session_start_time": session_start_time_from_meta,
        "weight_before_session": weight_before_session,
        "weight_after_session": weight_after_session,
        "water_during_session": weight_after_session - weight_before_session,
        "water_extra": extra_water,
    }

    df_session = pd.DataFrame(
        dict_meta,
        index=session_index,
    )
    # Use hierarchical index (type = {'metadata', 'session_stats'}, variable = {...}, etc.)
    df_session.columns = pd.MultiIndex.from_product(
        [["metadata"], dict_meta.keys()], names=["type", "variable"]
    )



    # -- Merge to df_session --
    df_session = pd.concat([df_session, df_session_stat], axis=1)
    return df_session


def create_df_session(nwb_filename):
    """
    Creates a dataframe where each row is a session
    nwb_filename can be either a single nwb file, a single filepath
    or a list of nwb files, or a list of nwb filepaths
    """
    if (type(nwb_filename) is not str) and (hasattr(nwb_filename, "__iter__")):
        dfs = []
        for nwb_file in nwb_filename:
            dfs.append(create_single_df_session(nwb_file))
        return pd.concat(dfs)
    else:
        return create_single_df_session(nwb_filename)


# % Process nwb and create df_session for every single session
def create_single_df_session(nwb_filename):
    """
    create a dataframe for a single session
    """
    nwb = load_nwb_from_filename(nwb_filename)

    df_session = create_single_df_session_inner(nwb)

    df_session.columns = df_session.columns.droplevel("type")
    df_session = df_session.reset_index()
    df_session["ses_idx"] = (
        df_session["subject_id"].values + "_" + df_session["session_date"].values
    )
    df_session = df_session.rename(columns={"variable": "session_num"})
    return df_session


def create_df_trials(nwb_filename, adjust_time=True, verbose=True):  # NOQA C901
    """
    Process nwb and create df_trials for every single session

    ARGS:
    nwb_filename (str or NWB object), the session to extract the trials from
    adjust_time (bool) if true, adjust t0 to be the first gocue

    RETURNS:
    A pandas dataframe containing the columns of nwb.trials plus:
    "_in_trial" time alignments where time is relative to the go cue on that trial
    "_in_session" time alignments where time is relative to the first go cue
        of the session.
    earned_reward, (bool) whether a reward was earned in that trial
    extra_reward (bool) whether a manual reward was given in that trial
    """

    # If we are given a filename, load the NWB object itself
    nwb = load_nwb_from_filename(nwb_filename)

    # Parse subject and session_date
    if nwb.session_id.startswith("behavior") or nwb.session_id.startswith("FIP"):
        splits = nwb.session_id.split("_")
        subject_id = splits[1]
        session_date = splits[2]
    else:
        splits = nwb.session_id.split("_")
        subject_id = splits[0]
        session_date = splits[1]
    ses_idx = subject_id + "_" + session_date

    # Build dataframe
    df = nwb.trials.to_dataframe().reset_index()
    df = df.rename(columns={"id": "trial"})
    df["ses_idx"] = ses_idx

    # Adjust for gaps in trial start/stop, and use the last stop time
    last_stop = df.iloc[-1]["stop_time"]
    df["stop_time"] = df["start_time"].shift(-1, fill_value=last_stop)


    # compute times relative to start of trial and start of session
    t0 = nwb.trials[SESSION_ALIGNMENT][0]
    drop_cols = []
    for col in df.columns:
        if ("time" in col):
            # Adjust all times relative to start of the first go cue
            if adjust_time:
                df[col + "_in_session"] = (df[col] - t0) / MS_TO_S
            else:
                df[col + "_in_session"] = df[col] / MS_TO_S

            # Adjust times relative to go cue on each trial
            if ("time" in col):
                # Here we always align to goCue_start_time, not SESSION_ALIGNMENT
                # since this aligns events relative to the trial go cue, not the start
                # of the session
                df[col + "_in_trial"] = (df[col].values - df["CS_start_time"].values) / MS_TO_S

            # Clean up these column names that are not clear
            drop_cols.append(col)

    # Add a column of raw time so users can map if they want
    raw_timepoints = ["start_time", "stop_time", "CS_start_time"]

    
    df = df.rename(columns = {raw_timepoint : raw_timepoint + '_raw' for raw_timepoint in raw_timepoints})
    


    return df


def create_df_events(nwb_filename, adjust_time=True, verbose=True):
    """
    returns a tidy dataframe of the events in the nwb file

    adjust_time (bool), set time of first goCue to t=0
    verbose (bool), give warnings for adjustments
    """

    nwb = load_nwb_from_filename(nwb_filename)

    # Build list of all event types in acqusition, ignore FIP events (no need for processing folder)
    # event_types = set(nwb.acquisition.keys())

    # channels = ["G", "R", "Iso"]
    # fibers = ["0", "1", "2", "3", "4"]
    # FIP_prefixes = [f"{c}_{f}" for c in channels for f in fibers]

    # # Filter out all fibers
    # event_types = {
    #     k for k in event_types if not any(k.startswith(prefix) for prefix in FIP_prefixes)
    # }

    # Version 1 pav nwb has pavlovian evnets in pavlovian event table
    if 'pavlovian_events' in nwb.processing:
        df = nwb.processing['pavlovian_events']['pavlovian_events_table'].to_dataframe().reset_index()

        df['timestamp_raw'] = df['timestamp']
        df['timestamp'] = df['timestamp'] / MS_TO_S

    # Determine time 0 as first go Cue
    if adjust_time:
        t0 = nwb.trials[SESSION_ALIGNMENT][0] / MS_TO_S
        df['timestamp'] = df['timestamp'] - t0
    else:
        t0 = 0

    # # Iterate over event types and build a dataframe of each
    # events = []
    # for e in event_types:
    #     # For each event, get timestamps, data, and label
    #     raw_stamps = nwb.acquisition[e].timestamps[:]
    #     data = nwb.acquisition[e].data[:]
    #     labels = [e] * len(data)
    #     stamps = raw_stamps - t0
    #     df = pd.DataFrame(
    #         {"timestamps": stamps, "data": data, "event": labels, "raw_timestamps": raw_stamps}
    #     )
    #     events.append(df)

    # # Add keys from trials table
    # trial_events = ["goCue_start_time"]
    # for e in trial_events:
    #     raw_stamps = nwb.trials[:][e].values
    #     labels = [e] * len(raw_stamps)
    #     data = [1] * len(raw_stamps)
    #     stamps = raw_stamps - t0
    #     df = pd.DataFrame(
    #         {"timestamps": stamps, "data": data, "event": labels, "raw_timestamps": raw_stamps}
    #     )
    #     events.append(df)

    # # Build dataframe by concatenating each event
    # df = pd.concat(events)
    # df = df.sort_values(by="timestamps")
    # df = df.dropna(subset="timestamps").reset_index(drop=True)


    # Sanity check that the first go cue is time 0
    # gocues = df.query("event == @SESSION_ALIGNMENT")
    # if (len(gocues) > 0) and (adjust_time):
    #     assert np.isclose(gocues.iloc[0]["timestamps"], 0, rtol=0.01)
    # # TODO, need more checks here for time alignment on trial index.

    if adjust_time and verbose:
        print(
            "Timestamps are adjusted such that `_in_session` timestamps start at the first go cue"
        )
    return df


def create_df_fip(nwb_filename, tidy=True, adjust_time=True, verbose=True):
    """
    returns a dataframe of the FIB data in the nwb file
    if tidy, return a tidy dataframe
    if not tidy, return pivoted by timestamp

    adjust_time (bool), set time of first goCue to t=0
    """

    nwb = load_nwb_from_filename(nwb_filename)

    # Build list of all FIB events in NWB file
    nwb_data = nwb.acquisition
    if len(nwb.processing):
        nwb_data = nwb.acquisition | nwb.processing["fiber_photometry"].data_interfaces

    event_types = set(nwb_data.keys())

    channels = ["G", "R", "Iso"]
    fibers = ["0", "1", "2", "3", "4"]
    FIP_prefixes = [f"{c}_{f}" for c in channels for f in fibers]

    # Filter out all fibers
    event_types = {k for k in event_types if any(k.startswith(prefix) for prefix in FIP_prefixes)}

    # If no FIB data available
    if len(event_types) == 0:
        return None

    # Determine time 0 as first go Cue
    if adjust_time:
        t0 = nwb.trials[SESSION_ALIGNMENT][0]
    else:
        t0 = 0

    # Iterate over event types and build a dataframe of each
    events = []
    for e in event_types:
        # For each event, get timestamps, data, and label
        raw_stamps = nwb_data[e].timestamps[:]
        data = nwb_data[e].data[:]
        labels = [e] * len(data)
        stamps = raw_stamps - t0
        df = pd.DataFrame(
            {"timestamps": stamps, "data": data, "event": labels, "raw_timestamps": raw_stamps}
        )
        events.append(df)

    # Build dataframe by concatenating each event
    df = pd.concat(events)
    df = df.sort_values(by="timestamps")
    df = df.dropna(subset="timestamps").reset_index(drop=True)

    # Add session_idx with subject ID and session date info - JL
    if nwb.session_id.startswith("behavior") or nwb.session_id.startswith("FIP"):
        splits = nwb.session_id.split("_")
        subject_id = splits[1]
        session_date = splits[2]
    else:
        splits = nwb.session_id.split("_")
        subject_id = splits[0]
        session_date = splits[1]
    ses_idx = subject_id + "_" + session_date
    df["ses_idx"] = ses_idx

    if adjust_time and verbose:
        print(
            "Timestamps are adjusted such that `_in_session` timestamps start at the first go cue"
        )
    df['timestamps'] = df['timestamps'] / MS_TO_S
    # pivot table based on timestamps
    if not tidy:
        df_pivoted = pd.pivot(df, index="timestamps", columns=["event"], values="data")
        return df_pivoted
    else:
        return df
