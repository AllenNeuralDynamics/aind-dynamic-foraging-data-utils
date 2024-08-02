"""Utility functions for processing dynamic foraging data."""

import re

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

LEFT, RIGHT = 0, 1


def foraging_eff_no_baiting(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):
    # Calculate foraging efficiency (only for 2lp)
    # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
    for_eff_optimal = reward_rate / np.nanmean(np.max([p_Ls, p_Rs], axis=0))
    if random_number_L is None:
        return for_eff_optimal, np.nan
    # --- Optimal-actual (uses the actual random numbers by simulation)
    reward_refills = np.vstack([p_Ls >= random_number_L, p_Rs >= random_number_R])
    # Greedy choice, assuming the agent knows the groundtruth
    optimal_choices = np.argmax([p_Ls, p_Rs], axis=0)
    optimal_rewards = (
        reward_refills[0][optimal_choices == 0].sum()
        + reward_refills[1][optimal_choices == 1].sum()
    )
    for_eff_optimal_random_seed = reward_rate / (optimal_rewards / len(optimal_choices))
    return for_eff_optimal, for_eff_optimal_random_seed


def foraging_eff_baiting(
    reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None
):  # Calculate foraging efficiency (only for 2lp)
    # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
    p_stars = np.zeros_like(p_Ls)
    for i, (p_L, p_R) in enumerate(zip(p_Ls, p_Rs)):  # Sum over all ps
        p_max = np.max([p_L, p_R])
        p_min = np.min([p_L, p_R])
        if p_min == 0 or p_max >= 1:
            p_stars[i] = p_max
        else:
            m_star = np.floor(np.log(1 - p_max) / np.log(1 - p_min))
            p_stars[i] = p_max + (1 - (1 - p_min) ** (m_star + 1) - p_max**2) / (m_star + 1)

    for_eff_optimal = reward_rate / np.nanmean(p_stars)

    if random_number_L is None:
        return for_eff_optimal, np.nan

    # --- Optimal-actual (uses the actual random numbers by simulation)
    block_trans = np.where(np.diff(np.hstack([np.inf, p_Ls, np.inf])))[0].tolist()
    reward_refills = [p_Ls >= random_number_L, p_Rs >= random_number_R]
    reward_optimal_random_seed = 0

    # Generate optimal choice pattern
    for b_start, b_end in zip(block_trans[:-1], block_trans[1:]):
        p_max = np.max([p_Ls[b_start], p_Rs[b_start]])
        p_min = np.min([p_Ls[b_start], p_Rs[b_start]])
        side_max = np.argmax([p_Ls[b_start], p_Rs[b_start]])

        # Get optimal choice pattern and expected optimal rate
        if p_min == 0 or p_max >= 1:
            this_choice = np.array([1] * (b_end - b_start))  # Greedy is obviously optimal
        else:
            m_star = np.floor(np.log(1 - p_max) / np.log(1 - p_min))
            this_choice = np.array(
                (([1] * int(m_star) + [0]) * (1 + int((b_end - b_start) / (m_star + 1))))[
                    : b_end - b_start
                ]
            )

        # Do simulation, using optimal choice pattern and actual random numbers
        reward_refill = np.vstack(
            [
                reward_refills[1 - side_max][b_start:b_end],
                reward_refills[side_max][b_start:b_end],
            ]
        ).astype(
            int
        )  # Max = 1, Min = 0
        reward_remain = [0, 0]
        for t in range(b_end - b_start):
            reward_available = reward_remain | reward_refill[:, t]
            reward_optimal_random_seed += reward_available[this_choice[t]]
            reward_remain = reward_available.copy()
            reward_remain[this_choice[t]] = 0

        if reward_optimal_random_seed:
            for_eff_optimal_random_seed = reward_rate / (reward_optimal_random_seed / len(p_Ls))
        else:
            for_eff_optimal_random_seed = np.nan

    return for_eff_optimal, for_eff_optimal_random_seed


def nwb_to_df(nwb):
    df_trials = nwb.trials.to_dataframe()

    # Reformat data
    choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
    p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])

    # -- Session-based table --
    # - Meta data -
    session_start_time_from_meta = nwb.session_start_time
    session_date_from_meta = session_start_time_from_meta.strftime("%Y-%m-%d")
    subject_id_from_meta = nwb.subject.subject_id

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
    if subject_id in ("689727"):
        subject_id_from_meta = subject_id

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

    dict_meta = {
        "rig": rig,
        "user_name": nwb.experimenter[0],
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

    # - Compute session-level stats -
    # TODO: Ideally, all these simple stats could be computed in the GUI, and
    # the GUI sends a copy to the meta session.json file and to the nwb file as well.

    total_trials = len(df_trials)
    finished_trials = np.sum(~np.isnan(choice_history))
    reward_trials = np.sum(reward_history)

    reward_rate = reward_trials / finished_trials

    # TODO: add more stats
    # See code here: https://github.com/AllenNeuralDynamics/map-ephys/blob/
    # 7a06a5178cc621638d849457abb003151f7234ea/pipeline/foraging_analysis.py#L70C8-L70C8
    # early_lick_ratio =
    # double_dipping_ratio =
    # block_num
    # mean_block_length
    # mean_reward_sum
    # mean_reward_contrast
    # autowater_num
    # autowater_ratio
    #
    # mean_iti
    # mean_reward_sum
    # mean_reward_contrast
    # ...

    foraging_eff_func = (
        foraging_eff_baiting if "bait" in nwb.protocol.lower() else foraging_eff_no_baiting
    )
    foraging_eff, foraging_eff_random_seed = foraging_eff_func(
        reward_rate, p_reward[LEFT, :], p_reward[RIGHT, :]
    )

    # -- Add session stats here --
    dict_session_stat = {
        "total_trials": total_trials,
        "finished_trials": finished_trials,
        "finished_rate": finished_trials / total_trials,
        "ignore_rate": np.sum(np.isnan(choice_history)) / total_trials,
        "reward_trials": reward_trials,
        "reward_rate": reward_rate,
        "foraging_eff": foraging_eff,
        # TODO: add more stats here
    }

    # Generate df_session_stat
    df_session_stat = pd.DataFrame(dict_session_stat, index=session_index)
    df_session_stat.columns = pd.MultiIndex.from_product(
        [["session_stats"], dict_session_stat.keys()],
        names=["type", "variable"],
    )

    # -- Add automatic training --
    if "auto_train_engaged" in df_trials.columns:
        df_session["auto_train", "curriculum_name"] = (
            np.nan
            if df_trials.auto_train_curriculum_name.mode()[0] == "none"
            else df_trials.auto_train_curriculum_name.mode()[0]
        )
        df_session["auto_train", "curriculum_version"] = (
            np.nan
            if df_trials.auto_train_curriculum_version.mode()[0] == "none"
            else df_trials.auto_train_curriculum_version.mode()[0]
        )
        df_session["auto_train", "curriculum_schema_version"] = (
            np.nan
            if df_trials.auto_train_curriculum_schema_version.mode()[0] == "none"
            else df_trials.auto_train_curriculum_schema_version.mode()[0]
        )
        df_session["auto_train", "current_stage_actual"] = (
            np.nan
            if df_trials.auto_train_stage.mode()[0] == "none"
            else df_trials.auto_train_stage.mode()[0]
        )
        df_session["auto_train", "if_overriden_by_trainer"] = (
            np.nan
            if all(df_trials.auto_train_stage_overridden.isna())
            else df_trials.auto_train_stage_overridden.mode()[0]
        )

        # Add a flag to indicate whether any of the auto train settings were changed
        # during the training
        df_session["auto_train", "if_consistent_within_session"] = (
            len(df_trials.groupby([col for col in df_trials.columns if "auto_train" in col])) == 1
        )
    else:
        for field in [
            "curriculum_name",
            "curriculum_version",
            "curriculum_schema_version",
            "current_stage_actual",
            "if_overriden_by_trainer",
        ]:
            df_session["auto_train", field] = None

    # -- Merge to df_session --
    df_session = pd.concat([df_session, df_session_stat], axis=1)
    return df_session


# % Process nwb and create df_session for every single session
def create_df_session(nwb_filename):
    io = NWBHDF5IO(nwb_filename, mode="r")
    nwb = io.read()
    df_session = nwb_to_df(nwb)

    # subject_id = df_session.index[0][0]
    # session_date = df_session.index[0][1]
    # nwb_suffix = df_session.index[0][2]
    # session_id = (
    #     f'{subject_id}_{session_date}{f"_{nwb_suffix}" if nwb_suffix else ""}'
    # )

    df_session.columns = df_session.columns.droplevel("type")
    df_session = df_session.reset_index()
    df_session["ses_idx"] = (
        df_session["subject_id"].values + "_" + df_session["session_date"].values
    )
    df_session = df_session.rename(columns={"variable": "session_num"})
    return df_session


# % Process nwb and create df_trials for every single session
def create_df_trials(nwb_filename):
    key_from_acq = [
        "left_lick_time",
        "right_lick_time",
        "left_reward_delivery_time",
        "right_reward_delivery_time",
        "FIP_falling_time",
        "FIP_rising_time",
    ]
    io = NWBHDF5IO(nwb_filename, mode="r")
    nwb = io.read()
    subject_id, session_date, session_json_time = re.match(
        r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>.*))\.json",
        nwb.session_id,
    ).groups()
    ses_idx = subject_id + "_" + session_date

    df_ses_trials = nwb.trials.to_dataframe().reset_index()
    df_ses_trials = df_ses_trials.rename(columns={"id": "trial"})
    df_ses_trials["ses_idx"] = ses_idx

    t0 = df_ses_trials.start_time[0]
    absolute_time = df_ses_trials["goCue_start_time"] - t0
    for col in df_ses_trials.columns:
        if ("time" in col) and (col != "goCue_start_time"):
            df_ses_trials.loc[:, col] = (
                df_ses_trials[col].values - df_ses_trials["goCue_start_time"].values
            )
    df_ses_trials["goCue_time_absolute"] = absolute_time
    df_ses_trials["goCue_start_time"] = 0.0
    events_ses = {key: nwb.acquisition[key].timestamps[:] - t0 for key in key_from_acq}

    for event in [
        "left_lick_time",
        "right_lick_time",
        "left_reward_delivery_time",
        "right_reward_delivery_time",
    ]:
        event_times = events_ses[event]
        df_ses_trials[event] = df_ses_trials.apply(
            lambda x: np.round(
                event_times[
                    (event_times > (x["goCue_start_time"] + x["goCue_time_absolute"]))
                    & (event_times < (x["stop_time"] + x["goCue_time_absolute"]))
                ]
                - x["goCue_time_absolute"],
                4,
            ),
            axis=1,
        )

    df_ses_trials["reward_time"] = df_ses_trials.apply(
        lambda x: np.nanmin(
            np.concatenate(
                [
                    [np.nan],
                    x["right_reward_delivery_time"],
                    x["left_reward_delivery_time"],
                ]
            )
        ),
        axis=1,
    )
    df_ses_trials["choice_time"] = df_ses_trials.apply(
        lambda x: np.nanmin(np.concatenate([[np.nan], x["right_lick_time"], x["left_lick_time"]])),
        axis=1,
    )
    df_ses_trials["reward"] = df_ses_trials.rewarded_historyR.astype(
        int
    ) | df_ses_trials.rewarded_historyL.astype(int)
    df_ses_trials = df_ses_trials.drop(
        columns=[
            "left_lick_time",
            "right_lick_time",
            "left_reward_delivery_time",
            "right_reward_delivery_time",
        ]
    )
    return df_ses_trials
    # result_folder = os.path.join(save_folder_df, session_id)
    # os.makedirs(result_folder, exist_ok=True)

    # pickle_file_name = result_folder + '/' + f'{session_id}_trials.pkl'
    # pd.to_pickle(df_ses_trials, pickle_file_name)


def create_events_df(nwb):
    """
    returns a tidy dataframe of the events in the nwb file
    """

    # Build list of all event types in acqusition, ignore FIP events
    event_types = set(nwb.acquisition.keys())
    ignore_types = set(
        [
            "FIP_falling_time",
            "FIP_rising_time",
            "G_1",
            "G_1_preprocessed",
            "G_2",
            "G_2_preprocessed",
            "Iso_1",
            "Iso_1_preprocessed",
            "Iso_1",
            "Iso_1_preprocessed",
            "R_1",
            "R_1_preprocessed",
            "R_2",
            "R_2_preprocessed",
            "Iso_2",
            "Iso_2_preprocessed",
        ]
    )
    event_types -= ignore_types

    # Iterate over event types and build a dataframe of each
    events = []
    for e in event_types:
        # For each event, get timestamps, data, and label
        stamps = nwb.acquisition[e].timestamps[:]
        data = nwb.acquisition[e].data[:]
        labels = [e] * len(data)
        df = pd.DataFrame({"timestamps": stamps, "data": data, "event": labels})
        events.append(df)

    # Build dataframe by concatenating each event
    df = pd.concat(events).reset_index(drop=True)
    df = df.sort_values(by="timestamps")
    df = df.dropna(subset="timestamps")

    return df


def get_time_array(
    t_start, t_end, sampling_rate=None, step_size=None, include_endpoint=True
):  # NOQA E501
    """
    A function to get a time array between two specified timepoints at a defined sampling rate  # NOQA E501
    Deals with possibility of time range not being evenly divisible by desired sampling rate  # NOQA E501
    Uses np.linspace instead of np.arange given decimal precision issues with np.arange (see np.arange documentation for details)  # NOQA E501

    Parameters:
    -----------
    t_start : float
        start time for array
    t_end : float
        end time for array
    sampling_rate : float
        desired sampling of array
        Note: user must specify either sampling_rate or step_size, not both
    step_size : float
        desired step size of array
        Note: user must specify either sampling_rate or step_size, not both
    include_endpoint : Boolean
        Passed to np.linspace to calculate relative time
        If True, stop is the last sample. Otherwise, it is not included.
            Default is True

    Returns:
    --------
    numpy.array
        an array of timepoints at the desired sampling rate

    Examples:
    ---------
    get a time array exclusive of the endpoint
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=1,
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])


    get a time array inclusive of the endpoint
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=1,
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5, 1.0])


    get a time array where the range can't be evenly divided by the desired step_size
    in this case, the time array includes the last timepoint before the desired endpoint
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=0.75,
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])


    Instead of passing the step_size, we can pass the sampling rate
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=1,
        sampling_rate=2,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])
    """
    assert (
        sampling_rate is not None or step_size is not None
    ), "must specify either sampling_rate or step_size"  # NOQA E501
    assert (
        sampling_rate is None or step_size is None
    ), "cannot specify both sampling_rate and step_size"  # NOQA E501

    # value as a linearly spaced time array
    if not step_size:
        step_size = 1 / sampling_rate
    # define a time array
    n_steps = (t_end - t_start) / step_size
    if n_steps != int(n_steps):
        # if the number of steps isn't an int, that means it isn't possible
        # to end on the desired t_after using the defined sampling rate
        # we need to round down and include the endpoint
        n_steps = int(n_steps)
        t_end_adjusted = t_start + n_steps * step_size
        include_endpoint = True
    else:
        t_end_adjusted = t_end

    if include_endpoint:
        # add an extra step if including endpoint
        n_steps += 1

    t_array = np.linspace(t_start, t_end_adjusted, int(n_steps), endpoint=include_endpoint)

    return t_array


def slice_inds_and_offsets(
    data_timestamps,
    event_timestamps,
    time_window,
    sampling_rate=None,
    include_endpoint=False,
):  # NOQA E501
    """
    Get nearest indices to event timestamps, plus ind offsets (start:stop)
    for slicing out a window around the event from the trace.
    Parameters:
    -----------
    data_timestamps : np.array
        Timestamps of the datatrace.
    event_timestamps : np.array
        Timestamps of events around which to slice windows.
    time_window : list
        [start_offset, end_offset] in seconds
    sampling_rate : float, optional, default=None
        Sampling rate of the datatrace.
        If left as None, samplng rate is inferred from data_timestamps.

    Returns:
    --------
    event_indices : np.array
        Indices of events from the timestamps provided.
    start_ind_offset : int
    end_ind_offset : int
    trace_timebase :  np.array
    """
    if sampling_rate is None:
        sampling_rate = 1 / np.diff(data_timestamps).mean()

    event_indices = index_of_nearest_value(data_timestamps, event_timestamps)
    trace_len = (time_window[1] - time_window[0]) * sampling_rate
    start_ind_offset = int(time_window[0] * sampling_rate)
    end_ind_offset = int(start_ind_offset + trace_len) + int(include_endpoint)
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / sampling_rate

    return event_indices, start_ind_offset, end_ind_offset, trace_timebase


def index_of_nearest_value(data_timestamps, event_timestamps):
    """
    The index of the nearest sample time for each event time.

    Parameters:
    -----------
    sample_timestamps : np.ndarray of floats
        sorted 1-d vector of data sample timestamps.
    event_timestamps : np.ndarray of floats
        1-d vector of event timestamps.

    Returns:
    --------
    event_aligned_ind : np.ndarray of int
        An array of nearest sample time index for each event times.
    """
    insertion_ind = np.searchsorted(data_timestamps, event_timestamps)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = data_timestamps[insertion_ind] - event_timestamps
    ind_minus_one_diff = np.abs(
        data_timestamps[np.clip(insertion_ind - 1, 0, np.inf).astype(int)] - event_timestamps
    )

    event_indices = insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)
    return event_indices


def event_triggered_response(
    data,
    t,
    y,
    event_times,
    t_start=None,
    t_end=None,
    t_before=None,
    t_after=None,
    output_sampling_rate=None,
    include_endpoint=True,
    output_format="tidy",
    interpolate=True,
):  # NOQA E501
    """
    Slices a timeseries relative to a given set of event times
    to build an event-triggered response. From mindscope_utilities (commit af2b70a)

    For example, If we have data such as a measurement of neural activity
    over time and specific events in time that we want to align
    the neural activity to, this function will extract segments of the neural
    timeseries in a specified time window around each event.

    The times of the events need not align with the measured
    times of the neural data.
    Relative times will be calculated by linear interpolation.

    Parameters:
    -----------
    data: Pandas.DataFrame
        Input dataframe in tidy format
        Each row should be one observation
        Must contains columns representing `t` and `y` (see below)
    t : string
        Name of column in data to use as time data
    y : string
        Name of column to use as y data
    event_times: list or array of floats
        Times of events of interest.
        Values in column specified by `y` will be sliced and interpolated
            relative to these times
    t_start : float
        start time relative to each event for desired time window
        e.g.:   t_start = -1 would start the window 1 second before each
                t_start = 1 would start the window 1 second after each event
        Note: cannot pass both t_start and t_before
    t_before : float
        time before each of event of interest to include in each slice
        e.g.:   t_before = 1 would start the window 1 second before each event
                t_before = -1 would start the window 1 second after each event
        Note: cannot pass both t_start and t_before
    t_end : float
        end time relative to each event for desired time window
        e.g.:   t_end = 1 would end the window 1 second after each event
                t_end = -1 would end the window 1 second before each event
        Note: cannot pass both t_end and t_after
    t_after : float
        time after each event of interest to include in each slice
        e.g.:   t_after = 1 would start the window 1 second after each event
                t_after = -1 would start the window 1 second before each event
        Note: cannot pass both t_end and t_after
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
    include_endpoint : Boolean
        Passed to np.linspace to calculate relative time
        If True, stop is the last sample. Otherwise, it is not included.
            Default is True
    output_format : string
        'wide' or 'tidy' (default = 'tidy')
        if 'tidy'
            One column representing time
            One column representing event_number
            One column representing event_time
            One row per observation (# rows = len(time) x len(event_times))
        if 'wide', output format will be:
            time as indices
            One row per interpolated timepoint
            One column per event,
                with column names titled event_{EVENT NUMBER}_t={EVENT TIME}
    interpolate : Boolean
        if True (default), interpolates each response onto a common timebase
        if False, shifts each response to align indices to a common timebase

    Returns:
    --------
    Pandas.DataFrame
        See description in `output_format` section above

    Examples:
    ---------
    An example use case, recover a sinousoid from noise:

    First, define a time vector
    >>> t = np.arange(-10,110,0.001)

    Now build a dataframe with one column for time,
    and another column that is a noise-corrupted sinuosoid with period of 1
    >>> data = pd.DataFrame({
            'time': t,
            'noisy_sinusoid': np.sin(2*np.pi*t) + np.random.randn(len(t))*3
        })

    Now use the event_triggered_response function to get a tidy
    dataframe of the signal around every event

    Events will simply be generated as every 1 second interval
    starting at 0, since our period here is 1
    >>> etr = event_triggered_response(
            data,
            x = 'time',
            y = 'noisy_sinusoid',
            event_times = np.arange(100),
            t_start = -1,
            t_end = 1,
            output_sampling_rate = 100
        )
    Then use seaborn to view the result
    We're able to recover the sinusoid through averaging
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> fig, ax = plt.subplots()
    >>> sns.lineplot(
            data = etr,
            x='time',
            y='noisy_sinusoid',
            ax=ax
        )
    """
    # ensure that non-conflicting time values are passed
    assert (
        t_before is not None or t_start is not None
    ), "must pass either t_start or t_before"  # noqa: E501
    assert (
        t_after is not None or t_end is not None
    ), "must pass either t_start or t_before"  # noqa: E501

    assert (
        t_before is None or t_start is None
    ), "cannot pass both t_start and t_before"  # noqa: E501
    assert t_after is None or t_end is None, "cannot pass both t_after and t_end"  # noqa: E501

    if interpolate is False:
        assert (
            output_sampling_rate is None
        ), "if interpolation = False, the sampling rate of the input timeseries will be used. Do not specify output_sampling_rate"  # NOQA E501

    # assign time values to t_start and t_end
    if t_start is None:
        t_start = -1 * t_before
    if t_end is None:
        t_end = t_after

    # ensure that t_end is greater than t_start
    assert t_end > t_start, "must define t_end to be greater than t_start"

    if output_sampling_rate is None:
        # if sampling rate is None,
        # set it to be the mean sampling rate of the input data
        output_sampling_rate = 1 / np.diff(data[t]).mean()

    # if interpolate is set to True,
    # we will calculate a common timebase and
    # interpolate every response onto that timebase
    if interpolate:
        # set up a dictionary with key 'time' and
        t_array = get_time_array(
            t_start=t_start,
            t_end=t_end,
            sampling_rate=output_sampling_rate,
            include_endpoint=include_endpoint,
        )
        data_dict = {"time": t_array}

        # iterate over all event times
        data_reindexed = data.set_index(t, inplace=False)

        for event_number, event_time in enumerate(np.array(event_times)):
            # get a slice of the input data surrounding each event time
            data_slice = data_reindexed[y].loc[
                event_time + t_start : event_time + t_end
            ]  # noqa: E501

            # if the slice is empty, we will fill it with NaNs
            if len(data_slice) == 0:
                data_dict.update(
                    {
                        "event_{}_t={}".format(event_number, event_time): np.full(
                            len(t_array), np.nan
                        )
                    }
                )

            # update our dictionary to have a new key defined as
            # 'event_{EVENT NUMBER}_t={EVENT TIME}' and
            # a value that includes an array that represents the
            # sliced data around the current event, interpolated
            # on the linearly spaced time array

            else:
                data_dict.update(
                    {
                        "event_{}_t={}".format(event_number, event_time): np.interp(
                            data_dict["time"],
                            data_slice.index - event_time,
                            data_slice.values,
                        )
                    }
                )

        # define a wide dataframe as a dataframe of the above compiled dictionary  # NOQA E501
        wide_etr = pd.DataFrame(data_dict)

    # if interpolate is False,
    # we will calculate a common timebase and
    # shift every response onto that timebase
    else:
        (
            event_indices,
            start_ind_offset,
            end_ind_offset,
            trace_timebase,
        ) = slice_inds_and_offsets(  # NOQA E501
            np.array(data[t]),
            np.array(event_times),
            time_window=[t_start, t_end],
            sampling_rate=None,
            include_endpoint=True,
        )
        all_inds = event_indices + np.arange(start_ind_offset, end_ind_offset)[:, None]
        wide_etr = (
            pd.DataFrame(
                data[y].values.T[all_inds],
                index=trace_timebase,
                columns=[
                    "event_{}_t={}".format(event_index, event_time)
                    for event_index, event_time in enumerate(event_times)
                ],  # NOQA E501
            )
            .rename_axis(index="time")
            .reset_index()
        )

    if output_format == "wide":
        # return the wide dataframe if output_format is 'wide'
        return wide_etr.set_index("time")
    elif output_format == "tidy":
        # if output format is 'tidy',
        # transform the wide dataframe to tidy format
        # first, melt the dataframe with the 'id_vars' column as "time"
        tidy_etr = wide_etr.melt(id_vars="time")

        # add an "event_number" column that contains the event number
        tidy_etr["event_number"] = (
            tidy_etr["variable"].map(lambda s: s.split("event_")[1].split("_")[0]).astype(int)
        )

        # add an "event_time" column that contains the event time ()
        tidy_etr["event_time"] = tidy_etr["variable"].map(lambda s: s.split("t=")[1]).astype(float)

        # drop the "variable" column, rename the "value" column
        tidy_etr = tidy_etr.drop(columns=["variable"]).rename(columns={"value": y})
        # return the tidy event triggered responses
        return tidy_etr
