"""Utility functions for processing dynamic foraging data."""

import os
import re

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from hdmf_zarr import NWBZarrIO

LEFT, RIGHT = 0, 1


def foraging_eff_no_baiting(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):
    """
    Calculate foraging efficiency (only for 2lp)
    """
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


def foraging_eff_baiting(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):
    """
    Calculate foraging efficiency (only for 2lp)
    """
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
    """
    given a nwb file, output a tidy dataframe
    """
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


def load_nwb_from_filename(filename):
    """
    Load NWB from file, checking for HDF5 or Zarr
    if filename is not a string, then return the input, assuming its the NWB file already
    """

    if type(filename) is str:
        if os.path.isdir(filename):
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

def create_df_session(nwb_filename):
    if (type(nwb_filename) is not str) and (hasattr(nwb_filename, '__iter__')):
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


def create_df_trials(nwb_filename):
    """
    Process nwb and create df_trials for every single session
    """
    nwb = load_nwb_from_filename(nwb_filename)

    key_from_acq = [
        "left_lick_time",
        "right_lick_time",
        "left_reward_delivery_time",
        "right_reward_delivery_time",
        "FIP_falling_time",
        "FIP_rising_time",
    ]

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


def create_events_df(nwb_filename):
    """
    returns a tidy dataframe of the events in the nwb file
    """

    nwb = load_nwb_from_filename(nwb_filename)

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


