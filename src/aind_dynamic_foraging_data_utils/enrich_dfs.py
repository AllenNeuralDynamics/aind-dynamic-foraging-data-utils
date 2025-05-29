"""
Important utility functions for enriching the dataframes

    enrich_df_trials_fm
    get_df_fip_trials
    remove_tonic_df_fip
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore


def extract_time_window(x, timepoints, alignment_events, offsets, absolute_time, align_in_trial):
    """
    extract out time window around the offsets around go_cue.
    Args:
        x (pd.Series): row of trial data from df_trials
        timepoints (list): timestamps for a particular session
        alignment_events (list): 2 timepoints in trial to align the offsets to
        offsets (list): Offsets from the 2 timepoints to extract time window
        absolute_time (str): column name in df_trials that defines start of trial in session
        align_in_trial (Bool): return the time window in_trial (as opposed to in_session)
    Returns:
        window (list): list of timepoints given in_session or in_trial (if align_in_trial)
    """
    # t_start, t_end for the trial
    t_start = x[alignment_events[0]] + offsets[0] + x[absolute_time]
    t_end = x[alignment_events[1]] + offsets[1] + x[absolute_time]
    # window of timestamps in session time
    window = timepoints[(timepoints > t_start) & (timepoints < t_end)]
    return window - x[absolute_time] if align_in_trial else window


def extract_signal_window(x, timepoints, varpoints, alignment_events, offsets, absolute_time):
    """
    extract out signal window around the offsets around go_cue.
    Args:
        x (pd.Series): row of trial data from df_trials
        timepoints (list): timestamps for a particular session
        varpoints (list): signal for a particular session
        alignment_events (list): 2 timepoints in trial to align the offsets to
        offsets (list): Offsets from the 2 timepoints to extract time window
        absolute_time (str): column name in df_trials that defines start of trial in session
        align_in_trial (Bool): return the time window in_trial (as opposed to in_session)
    Returns:
        window (list): list of timepoints given in_session or in_trial (if align_in_trial)
    """
    # t_start, t_end for the trial
    t_start = x[alignment_events[0]] + offsets[0] + x[absolute_time]
    t_end = x[alignment_events[1]] + offsets[1] + x[absolute_time]
    return varpoints[(timepoints > t_start) & (timepoints < t_end)]


def get_df_fip_trials(df_fip, df_trials):
    """
    Processes df_fip and df_trials, computing z-scored data and aligning timestamps.
    Args:
        df_fip (pd.DataFrame): A DataFrame with FIP data, each row a timepoint for a signal
        df_trials (pd.DataFrame): A DataFrame of trials (each row a trial)
    Returns:
        Tuple (df_fip, df_trials) with updated trial alignment and z-scored data.
    """

    # get the goCue of the next trial defined
    last_timestamps = df_fip.groupby('ses_idx', sort=False)['timestamps'].last()
    df_trials['goCue_start_time_next_in_session'] = df_trials.groupby('ses_idx')["goCue_start_time_in_session"].shift(-1, fill_value=-1)  # noqa: E501
    df_trials.loc[df_trials['goCue_start_time_next_in_session'] == -1, 'goCue_start_time_next_in_session'] = last_timestamps   # noqa: E501
    df_trials['goCue_start_time_next_in_trial'] = df_trials['goCue_start_time_next_in_session'] - df_trials['goCue_start_time_in_session']  # noqa: E501
    # set alignment events and offsets
    alignment_events = ['goCue_start_time_in_trial', 'goCue_start_time_next_in_trial']
    offsets = [-1, -1]

    df_fip.loc[:, 'data_z'] = df_fip.groupby(['ses_idx', 'event'])['data'].transform(
                                                 lambda x: zscore(x, ddof=1, nan_policy='omit'))
    for (ses_idx, event), df_fip_i in df_fip.groupby(['ses_idx', 'event']):
        # pull fip data into df_trials
        df_trials_ses = df_trials.loc[df_trials['ses_idx'] == ses_idx, :]
        if len(df_trials_ses) == 0:
            continue
        timepoints = df_fip_i['timestamps'].values
        absolute_time = 'goCue_start_time_in_session'
        df_trials.loc[df_trials['ses_idx'] == ses_idx, f'timestamps_in_trial_{event}'] = (
            df_trials_ses.apply(
                lambda x: extract_time_window(
                    x,
                    timepoints,
                    alignment_events,
                    offsets,
                    absolute_time,
                    align_in_trial=True
                ),
                axis=1
            )
        )
        df_trials.loc[df_trials['ses_idx'] == ses_idx, f'timestamps_in_session_{event}'] = (
            df_trials_ses.apply(
                lambda x: extract_time_window(
                    x,
                    timepoints,
                    alignment_events,
                    offsets,
                    absolute_time,
                    align_in_trial=False
                ),
                axis=1
            )
        )
        for var_event in ['data', 'data_z']:
            varpoints = df_fip_i[var_event].values
            df_trials.loc[df_trials['ses_idx'] == ses_idx, f'{var_event}_{event}'] = (
                df_trials_ses.apply(
                    lambda x: extract_signal_window(x, timepoints, varpoints,
                                                    alignment_events, offsets, absolute_time),
                    axis=1
                )
            )
    return (df_fip, df_trials)


def tidy_df_trials(df_fip, df_trials, col_prefix_signal):
    """
    Converts df_trials into a fully tidy long-format DataFrame.
    - Extracts event names from `timestamps_{event}` columns.
    - Keeps only `data_{event}_norm` and `data_z_{event}_norm` columns.
    - Explodes multiple combinations of timestamps and data columns.
    - Merges the tidied data with df_fip on `ses_idx` and `timestamps`.

    Args:
        df_fip (pd.DataFrame): A DataFrame with FIP data, each row a timepoint for a signal
        df_trials (pd.DataFrame): A DataFrame of trials (each row a trial)
        col_prefix_signal (str, optional): The prefix of the signal column name. Defaults to 'data'.

    Returns:
        df_tidy (pd.DataFrame): A long-format DataFrame with merged `df_fip`.
    """
    # Step 1: Identify relevant timestamp and data columns
    timestamp_cols = [col for col in df_trials.columns
                      if col.startswith("timestamps_in_session")]
    data_cols = [col for col in df_trials.columns
                 if col.startswith("data_") and col.endswith("_norm")]

    # Step 2: Initialize an empty list to store exploded DataFrames
    exploded_dfs = []

    # Step 3: Iterate over each timestamp-data column pair and explode them
    for timestamp_col in timestamp_cols:
        event = timestamp_col.replace("timestamps_in_session_", "")
        # Find matching data columns for this event
        matching_data_cols = [col for col in data_cols if event in col]

        # Create df_subset with timestamps and all matching data columns
        df_subset = df_trials[['ses_idx', timestamp_col] + matching_data_cols].copy()
        df_subset = df_subset.rename(columns={timestamp_col: 'timestamps'})
        df_subset['event'] = event  # Add event column

        # Rename data columns dynamically by removing the event part
        for data_col in matching_data_cols:
            new_data_col = data_col.replace(f"_{event}_norm", "_norm")
            new_data_col = data_col.replace(f"_{event}", "")
            df_subset = df_subset.rename(columns={data_col: new_data_col})
        df_subset = df_subset.dropna()

        # Explode timestamps and data together
        selected_cols = ['timestamps'] + [
            col for col in df_subset.columns if col.startswith(col_prefix_signal)
        ]
        df_subset = df_subset.explode(selected_cols, ignore_index=True)

        # Append to list for later concatenation
        exploded_dfs.append(df_subset)

    # Step 4: Concatenate all exploded DataFrames **before merging**
    df_exploded = pd.concat(exploded_dfs, ignore_index=True)

    # Step 5: Merge with df_fip on 'ses_idx' and 'timestamps'
    df_tidy = df_fip.merge(df_exploded, on=['ses_idx', 'timestamps', 'event'], how='left')
    return df_tidy.dropna().reset_index()


def remove_tonic_df_fip(df_fip, df_trials_fip, col_prefix_signal='data', col_prefix_time='timestamps_in_trial',
                        offset_from_goCue=[-1, 0], tidy=True):
    """
    Removes tonic activity by normalizing signal data against baseline.

    Args:
        df_fip (pd.DataFrame): A DataFrame with FIP data, each row a timepoint for a signal
        df_trials_fip (pd.DataFrame): A DataFrame of trials with signal data per trial
        col_prefix_signal (str, optional): The prefix of the signal column name. Defaults to 'data'.
        col_prefix_time (str, optional): The prefix of the time column name. Defaults to 'timestamps_in_trial_'.
        offset_from_goCue (list, optional): The baseline time range for normalization
                                    Time range is given as offsets from goCue of current trial to
                                    goCue of next trial. Defaults to [-1, 0].
        tidy (bool, optional): Whether to return a tidy DataFrame. Defaults to True.

    Returns:
        df_fip_tonic (pd.DataFrame): Updated df_fip with the baseline activity removed as columns
                                'data_{signal}_norm'. Averaged baseline is recorded as
                                'data_{signal}_baseline'
    """
    # Step 1: Find all columns with data signals.
    # Skip any columns that already have baseline removed.
    col_signals = [col for col in df_trials_fip.columns if col.startswith(col_prefix_signal)
                   and not col.endswith('_baseline')
                   and not col.endswith('_norm')]

    # Step 2: Loop through signal by signal, average the signal,
    # then remove the average activity per trial.
    for col_signal in col_signals:
        if col_signal + '_norm' in df_trials_fip.columns:
            continue
        signal_name = col_signal.removeprefix('data_z').removeprefix('data').removesuffix('_norm')
        df_trials_fip.loc[:, col_signal+'_baseline'] = df_trials_fip.apply(
            lambda x: np.nanmean(x[col_signal][(x[col_prefix_time+signal_name] < offset_from_goCue[1]) &
                                               (x[col_prefix_time+signal_name] > offset_from_goCue[0])])
            # Skip calculation if col_signal is NaN
            if not np.isnan(x[col_signal]).all() else np.nan,
            axis=1
        )
        df_trials_fip.loc[:, col_signal+'_norm'] = df_trials_fip.apply(
            lambda x: x[col_signal] - x[col_signal+'_baseline']
            # Skip calculation if col_signal is NaN
            if not np.isnan(x[col_signal]).all() else np.nan,
            axis=1
        )
    if tidy:
        return tidy_df_trials(df_fip, df_trials_fip, col_prefix_signal)
    return df_trials_fip
