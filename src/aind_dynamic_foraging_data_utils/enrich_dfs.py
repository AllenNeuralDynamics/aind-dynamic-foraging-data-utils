"""
Important utility functions for enriching the dataframes

    enrich_df_trials_fm
    get_df_fip_trials
"""

import numpy as np
import pandas as pd
import itertools


def get_df_fip_trials(input_obj):
    """
    Processes df_fip and df_trials, computing z-scored data and aligning timestamps.

    Args:
        input_obj: Either (df_fip, df_trials) OR an `nwb` object with attributes `df_fip` and `df_trials`.

    Returns:
        Tuple (df_fip, df_trials) with updated trial alignment and z-scored data.
    """
    
    # Extract df_fip and df_trials from input (handle both tuple and nwb object)
    if isinstance(input_obj, tuple):
        df_fip, df_trials = input_obj
    else:
        df_fip, df_trials = input_obj.df_fip, input_obj.df_trials


    # zscore data

    if 'data_z' not in df_fip.columns:
        df_fip.loc[:,'data_z'] = df_fip.groupby(['ses_idx', 'event'])['data'].transform(lambda x: zscore(x, nan_policy='omit'))


    for (ses_idx, event), df_fip_i in df_fip.groupby(['ses_idx', 'event']):
        # pull fip data into df_trials 

        df_trials_ses = df_trials.loc[df_trials['ses_idx'] == ses_idx,:]
        if len(df_trials_ses) == 0:
            continue

        timepoints = df_fip_i['timestamps'].values
        alignment_events = ['goCue_start_time_in_trial', 'stop_time_in_trial']
        offsets = [-1, 0]
        absolute_time = 'goCue_start_time_in_session'

        df_trials.loc[df_trials['ses_idx'] == ses_idx,
                                        f'timestamps_{event}'] =  df_trials_ses.apply(lambda x: timepoints[
                                        (timepoints>(x[alignment_events[0]]+offsets[0]+x[absolute_time])) &
                                        (timepoints<(x[alignment_events[1]]+offsets[1]+x[absolute_time]))] - x[absolute_time],
                                        axis=1)    
        for var_event in ['data', 'data_z']:
            varpoints = df_fip_i[var_event].values
            df_trials.loc[df_trials['ses_idx'] == ses_idx, 
                                        f'{var_event}_{event}'] = df_trials_ses.apply(lambda x: varpoints[
                                        (timepoints>(x[alignment_events[0]]+offsets[0]+x[absolute_time])) & 
                                        (timepoints<(x[alignment_events[1]]+offsets[1]+x[absolute_time]))],
                                        axis=1)
    return (df_fip, df_trials)



def tidy_df_trials(input_obj):
    """
    Converts df_trials into a fully tidy long-format DataFrame.
    - Extracts event names from `timestamps_{event}` columns.
    - Keeps only `data_{event}_norm` and `data_z_{event}_norm` columns.
    - Explodes multiple combinations of timestamps and data columns.
    - Merges the tidied data with df_fip on `ses_idx` and `timestamps`.

    Args:
        input_obj: Either (df_trials, df_fip) OR an `nwb` object with attributes `df_trials` and `df_fip`.

    Returns:
        pd.DataFrame: A long-format DataFrame with merged `df_fip`.
    """

    # Extract df_trials and df_fip from input (handle both tuple and nwb object)
    if isinstance(input_obj, tuple):
        df_trials, df_fip = input_obj
    else:
        df_trials, df_fip = input_obj.df_trials, input_obj.df_fip
        
    # Step 1: Identify relevant timestamp and data columns
    timestamp_cols = [col for col in df_trials.columns if col.startswith("timestamps_")]
    data_cols = [col for col in df_trials.columns if col.startswith("data_") and col.endswith("_norm")]

    # Step 2: Initialize an empty list to store exploded DataFrames
    exploded_dfs = []

    # Step 3: Iterate over each timestamp-data column pair and explode them
    for timestamp_col in timestamp_cols:
        event = timestamp_col.replace("timestamps_", "")  # Extract event name

        # correct timestamp from _in_trial to _in_session
        df_trials[timestamp_col] = df_trials.apply(
            lambda row: np.array(row[timestamp_col]) + row['goCue_start_time_in_session'], axis=1
        )
        # Find matching data columns for this event
        matching_data_cols = [col for col in data_cols if event in col]

        # Create df_subset with timestamps and all matching data columns
        df_subset = df_trials[['ses_idx', timestamp_col] + matching_data_cols].copy()
        df_subset = df_subset.rename(columns={timestamp_col: 'timestamps'})  # Rename timestamps column
        df_subset['event'] = event  # Add event column

        # Rename data columns dynamically by removing the event part
        for data_col in matching_data_cols:
            new_data_col = data_col.replace(f"_{event}_norm", "_norm")
            df_subset = df_subset.rename(columns={data_col: new_data_col})  

        # Explode timestamps and data together
        df_subset = df_subset.explode(['timestamps', 'data_norm', 'data_z_norm'], ignore_index=True)

        # Append to list for later concatenation
        exploded_dfs.append(df_subset)

    # Step 4: Concatenate all exploded DataFrames **before merging**
    df_exploded = pd.concat(exploded_dfs, ignore_index=True)

    # Step 5: Merge with df_fip on 'ses_idx' and 'timestamps', keeping all other columns from df_exploded
    df_fip_tonic = df_fip.merge(df_exploded, on=['ses_idx', 'timestamps', 'event'], how='left')

    return df_fip_tonic.dropna().reset_index()


def remove_tonic_df_fip(input_obj, col_signal='data', col_time='timestamps', baseline=[-1, 0], tidy = True):
    """
    Removes tonic activity by normalizing signal data against baseline.

    Args:
        input_obj: Either (df_trials_fip) OR an `nwb` object with attribute `df_trials_fip`.
                    nwb object is the nwb read with NWBZarrIO and a df_trials_fip (using get_df_trials_fip)
                    df_trials_fip is a dataframe with of trials with signal data per trial 
        col_signal (str, optional): The name of the signal column. Defaults to 'data'.
        col_time (str, optional): The name of the time column. Defaults to 'timestamps'.
        baseline (list, optional): A list specifying the baseline time range for normalization. Defaults to [-1, 0].
        tidy (bool, optional): Whether to return a tidy DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: The modified DataFrame with additional computed columns.
    """

    # Extract df_trials and df_fip from input (handle both tuple and nwb object)
    if isinstance(input_obj, tuple):
        df_trials_fip, df_fip = input_obj
    else:
        df_trials_fip, df_fip = input_obj.df_trials, input_obj.df_fip

    col_signals = [col for col in df_trials_fip.columns if col.startswith(col_signal) 
                                                and not col.endswith('_baseline')
                                                and not col.endswith('_norm')]

    # for this to work, i need to match df_fip_trials to as close to the OUTPUT to def create_df_trials_events
    # check this against the format_data_OLD.ipynb and undersatnd why df_fip_trials works! 
    for col_signal in col_signals:
        if col_signal + '_norm' in df_trials_fip.columns:
            continue
        signal_name = col_signal.removeprefix('data_z').removeprefix('data').removesuffix('_norm')
        df_trials_fip.loc[:, col_signal+'_baseline'] = df_trials_fip.apply(lambda x: np.mean(
                                            x[col_signal][(x[col_time+signal_name]<baseline[1]) & 
                                                          (x[col_time+signal_name]>baseline[0])]), axis=1)

        df_trials_fip.loc[:, col_signal+'_norm'] = df_trials_fip.apply(lambda x: 
                                            x[col_signal]-x[col_signal+'_baseline'], axis=1)   

    if tidy:
        return tidy_df_trials((df_trials_fip, df_fip))
    return df_trials_fip