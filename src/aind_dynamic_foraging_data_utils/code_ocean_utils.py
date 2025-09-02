"""
Important utility functions for formatting the data
    get_subject_assets
    generate_data_asset_attach_params
    attach_data
    check_data_assets
    add_data_asset_path
    get_all_df_for_nwb
    get_foraging_model_info
"""

import os
import warnings

import numpy as np
import pandas as pd
from aind_analysis_arch_result_access.han_pipeline import get_mle_model_fitting
from aind_data_access_api.document_db import MetadataDbClient
from codeocean import CodeOcean
from codeocean.data_asset import DataAssetAttachParams

from aind_dynamic_foraging_data_utils import nwb_utils


def get_subject_assets(subject_id, **kwargs):
    """
    Returns the docDB results for a subject. If duplicate entries exist, take the last
    based on processing time. Skips pavlovian task.

    equivalent to get_assets(subjects=[subject_id])

    subject_id (str or int) subject id to get assets for from docDB
    processed (bool) if True, look for processed assets. If False, look for raw assets
    task (list of strings), if empty, include all task variants: Uncoupled Baiting,
        Coupled Baiting, Uncoupled Without Baiting, Coupled Without Baiting.
        If not empty, only include the task variants provided.
    modality (list of strings), required data modality. If empty list, does not filter
        modalities should the data modality abbreviations, for example: behavior,
        behavior-videos, fib, ecephys
    extra_filter (dict), docdb query

    Example
    results = get_subject_assets(my_id)
    co_assets = attach_data(results['code_ocean_asset_id'].values)

    To filter by stage:
    stage_filter = {
        "session.stimulus_epochs.output_parameters.task_parameters.stage_in_use":
        {"$regex" :"STAGE_FINAL"}
        }
    Note, until backfilling is done this information is missing in older data, see
    this issue for progress and details:
    https://github.com/AllenNeuralDynamics/aind-data-migration-scripts/issues/374

    """
    return get_assets(subjects=[subject_id], **kwargs)


def get_assets(subjects=[], processed=True, task=[], modality=["behavior"], extra_filter={}):
    """
    Returns the docDB results for a subject. If duplicate entries exist, take the last
    based on processing time. Skips pavlovian task.

    subjects (a list of strs or ints) subject ids to get assets for from docDB
    processed (bool) if True, look for processed assets. If False, look for raw assets
    task (list of strings), if empty, include all task variants: Uncoupled Baiting,
        Coupled Baiting, Uncoupled Without Baiting, Coupled Without Baiting.
        If not empty, only include the task variants provided.
    modality (list of strings), required data modality. If empty list, does not filter
        modalities should the data modality abbreviations, for example: behavior,
        behavior-videos, fib, ecephys
    extra_filter (dict), docdb query

    Example
    results = get_assets(subjects=[my_id])
    co_assets = attach_data(results['code_ocean_asset_id'].values)

    To filter by stage:
    stage_filter = {
        "session.stimulus_epochs.output_parameters.task_parameters.stage_in_use":
        {"$regex" :"STAGE_FINAL"}
        }
    Note, until backfilling is done this information is missing in older data, see
    this issue for progress and details:
    https://github.com/AllenNeuralDynamics/aind-data-migration-scripts/issues/374

    """
    # Create metadata client
    client = MetadataDbClient(
        host="api.allenneuraldynamics.org", database="metadata_index", collection="data_assets"
    )

    # Filter by task
    if len(task) > 0:
        task_filter = {"$or": []}
        for t in task:
            task_filter["$or"].append(
                {
                    "session.session_type": {"$regex": "^{}".format(t)},
                }
            )
    else:
        task_filter = {
            "session.session_type": {"$regex": "^(Uncoupled|Coupled)( Without)? Baiting"},
        }

    # Filter by data modality
    if len(modality) > 0:
        modality_filter = {"$and": []}
        for m in modality:
            modality_filter["$and"].append(
                {"data_description.modality.abbreviation": {"$regex": m}}
            )
    else:
        modality_filter = {}

    # Do we want processed or raw assets
    if processed:
        processed_string = "_.*processed_[0-9-_]*"
    else:
        processed_string = "_.*"

    # Query based on subject id
    if len(subjects) == 0:
        print("Query will be slow without explicit subject ids")
        subject_filter = {
            "name": {"$regex": "^behavior_[0-9]*{}$".format(processed_string)},
        }
        # Return only essential information for performance
        projection = {
            "name": 1,
            "_id": 1,
            "session": 1,
            "session_name": 1,
            "external_links": 1,
            "subject.subject_id": 1,
        }
    else:
        subject_filter = {
            "name": {
                "$regex": "^behavior_("
                + "|".join([str(x) for x in subjects])
                + "){}$".format(processed_string)
            },
        }
        # Return all information
        projection = None

    # Query
    results = pd.DataFrame(
        client.retrieve_docdb_records(
            filter_query={
                **subject_filter,
                **task_filter,
                **modality_filter,
                **extra_filter,
            },
            projection=projection,
        )
    )

    # If nothing is found, return
    if len(results) == 0:
        print("No results found for {}".format(subjects))
        return

    # look for duplicate entries, taking the last by processing time
    results["session_name"] = [x.split("_processed")[0] for x in results["name"]]
    results = results.sort_values(by="name")
    results_no_duplicates = results.drop_duplicates(subset="session_name", keep="last").copy()

    # If there were duplicates, make a warning and print the duplicates
    if len(results) != len(results_no_duplicates):
        duplicated = results[results.duplicated(subset="session_name", keep=False)]
        warnings.warn("Duplicate session entries in docDB")
        for index, row in duplicated.iterrows():
            print("duplicated: {}".format(row["name"]))

    # Make code ocean ID a column
    results_no_duplicates["code_ocean_asset_id"] = [
        link["Code Ocean"][0] if "Code Ocean" in link else ""
        for link in results_no_duplicates["external_links"]
    ]

    return results_no_duplicates.reset_index(drop=True)


def generate_data_asset_attach_params(data_asset_IDs, mount_point=None):
    """
    generate_data_asset_attach_params is a helper function for attach_data
    data_asset_IDs:  list of data asset IDs, i.e. the 16 hash string for the data asset in CO.
    mount_point: the mount point (folder) for the data asset. Default is None.
    """
    data_assets = []
    for ID in data_asset_IDs:
        if mount_point:
            data_assets.append(DataAssetAttachParams(id=ID, mount=mount_point))
        else:
            data_assets.append(DataAssetAttachParams(id=ID))
    return data_assets


def attach_data(data_asset_IDs, token_name="CUSTOM_KEY"):
    """
    attach_data attaches a list of data_asset_ID to the capsule.
    data_asset_IDs: list of data asset IDs, i.e. the 16 hash string for the data asset in CO.
    token_name: the name of the token in the environment variable. Default is CUSTOM_KEY.
                see more info here:
                https://docs.codeocean.com/user-guide/code-ocean-api/authentication#to-create-an-access-token

    Note that the list of data_asset_IDs should be <100 or you may risk CO crashing.
    Example
    results = get_subject_assets(my_id)
    co_assets = attach_data(results['code_ocean_asset_id'].values)
    """

    # Check for too many assets
    if len(data_asset_IDs) > 100:
        warnings.warn(
            "Attaching more than 100 data assets at one time may crash Code Ocean. "
            + "Please batch data into 100 sessions per call"
        )
        return

    # Get asset attach params
    data_assets = generate_data_asset_attach_params(data_asset_IDs, mount_point=None)

    # Configure api
    token = os.getenv(token_name)
    if not token:
        warnings.warn("no token found. please create token")
        return
    client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=token)
    capsule_id = os.getenv("CO_CAPSULE_ID")

    # Try to attach all assets
    try:
        results = client.capsules.attach_data_assets(
            capsule_id=capsule_id,
            attach_params=data_assets,
        )
        return results
    except Exception as e:
        print(e)
        print("Attaching assets in a batch failed, trying individually")

    # Try to attach assets one by one (slow)
    for asset in data_assets:
        try:
            results = client.capsules.attach_data_assets(
                capsule_id=capsule_id,
                attach_params=[asset],
            )
        except Exception as e:
            print("Could not attach this asset: {}".format(e.data[0]))
    return results


def check_data_assets(co_assets, data_asset_IDs):
    """
    co_assets: a list of DataAssetAttachResults, produced by attach_data()
    data_asset_IDs: a list of code ocean Data Assets, passed into attach_data()
    This function is delicate because CO is strange about "ready",
    but its a useful quick check
    """
    if all([x.ready for x in co_assets if x.id in data_asset_IDs]):
        print("all data assets are ready")
    else:
        print("some data assets are not ready")


def add_data_asset_path(results):
    """
    Adds the filepath to the dataframe
    Example
    results = get_subject_assets(my_id)
    results = add_data_asset_path(results)
    """
    results["data_path"] = [
        os.path.join("data", x["name"], "nwb", x["session_name"] + ".nwb")
        for index, x in results.iterrows()
    ]
    return results


def get_all_df_for_nwb(filename_sessions, interested_channels=None):
    """
    get_all_df_for_nwb gets all the dataframes for the NWB and
    returns the df_trials, df_events, df_fip

    Returns:
        df_trials (pd.DataFrame): dataframe with each row a trial from sessions
        df_events (pd.DataFrame): dataframe with each row an event from sessions
        df_fip (pd.DataFrame): dataframe with each row a timepoint for a signal from sessions
    """
    print(f"Saving channels: {interested_channels}")

    # Lists to collect session-level DataFrames
    list_df_trials = []
    list_df_events = []
    list_df_fip = []

    for idx, nwb_file in enumerate(filename_sessions):
        nwb = nwb_utils.load_nwb_from_filename(nwb_file)
        ses_idx = nwb_file.split("/")[-1].replace("behavior_", "").rsplit("_", 1)[0]
        print(f"CURRENTLY RUNNING {idx+1}/{len(filename_sessions)}: {ses_idx}")
        print("--------------------------------------------------")

        # Try to process df_trials, df_fip, df_events, otherwise go to next nwb_file

        # Trials
        try:
            df_ses_trials = nwb_utils.create_df_trials(nwb)
            list_df_trials.append(df_ses_trials)
        except AssertionError as e:
            print(f"Skipping {ses_idx} due to assertion error in df_trials: {e}")
            continue

        # FIP
        try:
            df_ses_fip = nwb_utils.create_fib_df(nwb, tidy=True)
            if interested_channels:
                df_ses_fip = df_ses_fip[df_ses_fip["event"].isin(interested_channels)]
            list_df_fip.append(df_ses_fip)
        except AssertionError as e:
            print(f"Skipping {ses_idx} due to assertion error in df_fip: {e}")
            continue

        # Events
        try:
            df_ses_events = nwb_utils.create_events_df(nwb)
            df_ses_events["ses_idx"] = ses_idx  # Add session identifier
            list_df_events.append(df_ses_events)
        except AssertionError as e:
            print(f"Skipping {ses_idx} due to assertion error in df_events: {e}")
            continue

    # Concatenate all collected DataFrames
    df_trials = pd.concat(list_df_trials, axis=0).reset_index(drop=True)
    df_events = pd.concat(list_df_events, axis=0).reset_index(drop=True)
    df_fip = pd.concat(list_df_fip, axis=0).reset_index(drop=True)

    return (df_trials, df_events, df_fip)


def get_foraging_model_info(
    df_trials, df_sess, loc=None, model_name="QLearning_L2F1_CKfull_softmax"
):
    """
    get_foraging_model_info: retrieves fitted foraging_model information
    df_trials: dataframe for trials (1 row per trials) from nwb_utils.create df_trials.
               saved df_trials_fm will have L_prob, R_prob, L_value, R_value,
               (if choice kernel in model), L_kernel, R_kernel
    df_sess: dataframe for sessions (1 row per session) from nwb_utils.create_df_sessions
               saved df_sess_fm will have parameters fitted for each mouse.
    loc: location to save the updated df_trials, df_sess with suffix `_fm.csv`.
                If given, we will save, otherwise we will return the dataframes
    model_name: model alias to get the model information

    """
    print(f"Retrieving foraging model {model_name}")
    df_trials_fm = df_trials.copy()
    df_trials_fm = df_trials_fm.rename(columns={"animal_response": "choice"})
    df_trials_fm["choice_name"] = df_trials_fm["choice"].map({1: "right", 0: "left"})

    df_trials_fm["model_name"] = model_name
    df_trials_fm["L_prob"] = np.nan
    df_trials_fm["R_prob"] = np.nan
    df_trials_fm["L_value"] = np.nan
    df_trials_fm["R_value"] = np.nan

    # check if CK is in model_name, if so, add df_trials_fm['L_kernel]
    if "CK" in model_name:
        df_trials_fm["L_kernel"] = np.nan
        df_trials_fm["R_kernel"] = np.nan

    df_sess_params = []
    for index, sess_i in df_sess.iterrows():
        df = get_mle_model_fitting(
            subject_id=str(sess_i["subject_id"]),
            session_date=str(sess_i["session_date"]),
            agent_alias=model_name,
        )

        if df is None:
            continue  # skip if no model fits is found for this session

        # Fitted parameters
        params = df["params"][0]

        # Fitted latent variables
        fitted_latent = df["latent_variables"][0]

        # pull the information
        mouse_choice_idx = df_trials_fm.index[
            (df_trials_fm["ses_idx"] == str(sess_i["ses_idx"])) & (df_trials_fm["choice"] < 2)
        ]

        qvals = np.array(fitted_latent["q_value"]).astype(float)
        choice_kernel = np.array(fitted_latent["choice_kernel"]).astype(float)
        choice_prob = np.array(fitted_latent["choice_prob"]).astype(float)

        df_trials_fm.loc[mouse_choice_idx, "L_prob"] = choice_prob[0, :]
        df_trials_fm.loc[mouse_choice_idx, "R_prob"] = choice_prob[1, :]
        df_trials_fm.loc[mouse_choice_idx, "L_value"] = qvals[0, :-1]
        df_trials_fm.loc[mouse_choice_idx, "R_value"] = qvals[1, :-1]

        if "CK" in model_name:
            df_trials_fm.loc[mouse_choice_idx, "L_kernel"] = choice_kernel[0, :-1]
            df_trials_fm.loc[mouse_choice_idx, "R_kernel"] = choice_kernel[1, :-1]

        params["ses_idx"] = str(sess_i["ses_idx"])
        df_sess_params.append(params)

    df_sess_params = pd.DataFrame(df_sess_params)
    df_sess_fm = df_sess.merge(df_sess_params, how="left", on=["ses_idx"])

    if loc:
        df_sess_fm.to_csv(loc + "df_sess_fm.csv")
        df_trials_fm.to_csv(loc + "df_trials_fm.csv")

    else:
        return df_trials_fm, df_sess_fm
