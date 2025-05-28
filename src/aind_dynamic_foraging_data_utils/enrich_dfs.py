"""
Important utility functions for enriching the dataframes

    enrich_df_trials_fm
    get_df_fip_trials
    remove_tonic_df_fip
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

def get_df_fip_trials(input_obj):
    # TODO:Alex commented 
    """
    I don't think we should support taking in a tuple as an argument. 
    I think we should be consistent about operating on an analysis object, or a dataframe, not a third option
    """
    """
    Processes df_fip and df_trials, computing z-scored data and aligning timestamps.
    Args:
        input_obj: Either (df_fip, df_trials) OR an `nwb` object
                    with attributes `df_fip` and `df_trials`.
    Returns:
        Tuple (df_fip, df_trials) with updated trial alignment and z-scored data.
    """

    # Extract df_fip and df_trials from input (handle both tuple and nwb object)
    if isinstance(input_obj, tuple):
        df_fip, df_trials = input_obj
    else:
        df_fip, df_trials = input_obj.df_fip, input_obj.df_trials
    # get the goCue of the next trial defined
    last_timestamps = df_fip.groupby('ses_idx', sort=False)['timestamps'].last()
    df_trials['goCue_start_time_next_in_session'] = df_trials.groupby('ses_idx')["goCue_start_time_in_session"].shift(-1, fill_value=-1)  # noqa: E501
    df_trials.loc[df_trials['goCue_start_time_next_in_session'] == -1, 'goCue_start_time_next_in_session'] = last_timestamps   # noqa: E501
    df_trials['goCue_start_time_next_in_trial'] = df_trials['goCue_start_time_next_in_session'] - df_trials['goCue_start_time_in_session']  # noqa: E501
    # set alignment events and offsets
    alignment_events = ['goCue_start_time_in_trial', 'goCue_start_time_next_in_trial']
    offsets = [-1, -1]
    # zscore data if needed
    if 'data_z' not in df_fip.columns:
        df_fip.loc[:, 'data_z'] = df_fip.groupby(['ses_idx', 'event'])['data'].transform(
                                                 lambda x: zscore(x, ddof=1, nan_policy='omit'))
    # TODO: ALex commented: 
    """
    I think I would enforce that z-scoring only happens in one place. 
    So if this is sometimes happening elsewhere, I would always do it there
    """
    for (ses_idx, event), df_fip_i in df_fip.groupby(['ses_idx', 'event']):
        # pull fip data into df_trials
        df_trials_ses = df_trials.loc[df_trials['ses_idx'] == ses_idx, :]
        if len(df_trials_ses) == 0:
            continue
        timepoints = df_fip_i['timestamps'].values
        absolute_time = 'goCue_start_time_in_session'
        df_trials.loc[df_trials['ses_idx'] == ses_idx, f'timestamps_{event}'] = df_trials_ses.apply(lambda x: timepoints[  # noqa: E501
                      (timepoints > (x[alignment_events[0]]+offsets[0]+x[absolute_time])) &
                      (timepoints < (x[alignment_events[1]]+offsets[1]+x[absolute_time]))]
                      - x[absolute_time], axis=1)
        df_trials.loc[df_trials['ses_idx'] == ses_idx, f'timestamps_{event}_in_session'] = df_trials_ses.apply(lambda x: timepoints[  # noqa: E501
                      (timepoints > (x[alignment_events[0]]+offsets[0]+x[absolute_time])) &
                      (timepoints < (x[alignment_events[1]]+offsets[1]+x[absolute_time]))],
                      axis=1)
        # TODO: Alex commented
        """
        this section is hard to parse, can you break it down into steps and add comments?
        """
        for var_event in ['data', 'data_z']:
            varpoints = df_fip_i[var_event].values
            df_trials.loc[df_trials['ses_idx'] == ses_idx, f'{var_event}_{event}'] = df_trials_ses.apply(lambda x: varpoints[  # noqa: E501
                          (timepoints > (x[alignment_events[0]]+offsets[0]+x[absolute_time])) &
                          (timepoints < (x[alignment_events[1]]+offsets[1]+x[absolute_time]))],
                          axis=1)
    return (df_fip, df_trials)


def tidy_df_trials(input_obj, test=False):
    """
    Converts df_trials into a fully tidy long-format DataFrame.
    - Extracts event names from `timestamps_{event}` columns.
    - Keeps only `data_{event}_norm` and `data_z_{event}_norm` columns.
    - Explodes multiple combinations of timestamps and data columns.
    - Merges the tidied data with df_fip on `ses_idx` and `timestamps`.

    Args:
        input_obj: Either (df_trials, df_fip) OR an `nwb` object
                    with attributes `df_trials` and `df_fip`.
        test: determines if we include 'data' and 'data_z' to test the merge

    Returns:
        pd.DataFrame: A long-format DataFrame with merged `df_fip`.
    """
    # TODO: Alex commented
    """
    Again, I don't think we should support a tuple as an input option
    """
    # Extract df_trials and df_fip from input (handle both tuple and nwb object)
    if isinstance(input_obj, tuple):
        df_trials, df_fip = input_obj
    else:
        df_trials, df_fip = input_obj.df_trials, input_obj.df_fip
# Step 1: Identify relevant timestamp and data columns
    timestamp_cols = [col for col in df_trials.columns
                      if col.startswith("timestamps_") and col.endswith("in_session")]
    # TODO: ALex commented:
    """
    Doesn't have to be in_session right? If the timestamps aren't aligned to the session start
    """
    if test:
        data_cols = [col for col in df_trials.columns
                     if col.startswith("data_") and not col.endswith("baseline")]
    else:
        data_cols = [col for col in df_trials.columns
                     if col.startswith("data_") and col.endswith("_norm")]

    # Step 2: Initialize an empty list to store exploded DataFrames
    exploded_dfs = []

    # Step 3: Iterate over each timestamp-data column pair and explode them
    for timestamp_col in timestamp_cols:
        event = timestamp_col.replace("timestamps_", "").replace("_in_session", "")
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
        # TODO: Alex commented-- All "TEST" should be taken out. 
        """
        you could make the data columns "data", "data_z" input arguments right? That way this function becomes more general
        """
        if test:
            df_subset = df_subset.explode(['timestamps', 'data', 'data_z',
                                           'data_norm', 'data_z_norm'], ignore_index=True)
        else:
            df_subset = df_subset.explode(['timestamps', 'data_norm',
                                           'data_z_norm'], ignore_index=True)

        # Append to list for later concatenation
        exploded_dfs.append(df_subset)

    # Step 4: Concatenate all exploded DataFrames **before merging**
    df_exploded = pd.concat(exploded_dfs, ignore_index=True)

    # Step 5: Merge with df_fip on 'ses_idx' and 'timestamps'
    df_fip_tonic = df_fip.merge(df_exploded, on=['ses_idx', 'timestamps', 'event'], how='left')
    # TODO: alex commmented
    """
    Why is this called tonic? This function isn't specific to tonic analysis, right?
    """

    return df_fip_tonic.dropna().reset_index()


def remove_tonic_df_fip(input_obj, col_signal='data', col_time='timestamps',
                        baseline=[-1, 0], tidy=True):
    """
    Removes tonic activity by normalizing signal data against baseline.

    Args:
        input_obj: Either (df_trials_fip) OR an `nwb` object with attribute `df_trials_fip`.
                    nwb object is the nwb read with NWBZarrIO and a df_trials_fip
                    (using get_df_trials_fip)
                    df_trials_fip is a dataframe with of trials with signal data per trial
        col_signal (str, optional): The name of the signal column. Defaults to 'data'.
        col_time (str, optional): The name of the time column. Defaults to 'timestamps'.
        baseline (list, optional): A list specifying the baseline time range for normalization.
                                    Defaults to [-1, 0].
        tidy (bool, optional): Whether to return a tidy DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: The modified DataFrame with additional computed columns.
    """
    # TODO: ALex commented
    """
  What are the additional columns named? Can you define the naming scheme?
        """
    # Extract df_trials and df_fip from input (handle both tuple and nwb object)
    if isinstance(input_obj, tuple):
        df_trials_fip, df_fip = input_obj
    else:
        df_trials_fip, df_fip = input_obj.df_trials, input_obj.df_fip

    col_signals = [col for col in df_trials_fip.columns if col.startswith(col_signal)
                   and not col.endswith('_baseline')
                   and not col.endswith('_norm')]
    for col_signal in col_signals:
        if col_signal + '_norm' in df_trials_fip.columns:
            continue
        signal_name = col_signal.removeprefix('data_z').removeprefix('data').removesuffix('_norm')
        df_trials_fip.loc[:, col_signal+'_baseline'] = df_trials_fip.apply(
            lambda x: np.nanmean(x[col_signal][(x[col_time+signal_name] < baseline[1]) &
                                               (x[col_time+signal_name] > baseline[0])])
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
        return tidy_df_trials((df_trials_fip, df_fip))
    return df_trials_fip
