"""
Important utility functions for formatting the data
    get_subject_assets
    attach_data
    get_all_df_for_nwb

    check_avail_model_by_nwb_name
    get_foraging_model_info
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import requests
from codeocean import CodeOcean
from codeocean.data_asset import DataAssetAttachParams
from aind_data_access_api.document_db import MetadataDbClient

from aind_dynamic_foraging_data_utils import nwb_utils

URL = "https://api.allenneuraldynamics-test.org/v1/behavior_analysis/mle_fitting"


def get_subject_assets(subject_id, processed=True):
    """
    Returns the docDB results for a subject. If duplicate entries exist, take the last
    based on processing time

    subject_id (str or int) subject id to get assets for from docDB
    processed (bool) if True, look for processed assets. If False, look for raw assets

    Example
    results = get_subject_assets(my_id)
    co_assets = attach_data(results['_id'].values)
    """

    # Create metadata client
    client = MetadataDbClient(
        host="api.allenneuraldynamics.org", database="metadata_index", collection="data_assets"
    )

    # Query based on subject id
    if processed:
        results = pd.DataFrame(
            client.retrieve_docdb_records(
                filter_query={
                    "name": {"$regex": "^behavior_{}_.*processed_[0-9-_]*$".format(subject_id)}
                }
            )
        )
    else:
        results = pd.DataFrame(
            client.retrieve_docdb_records(
                filter_query={"name": {"$regex": "^behavior_{}_[0-9-_]*$".format(subject_id)}}
            )
        )

    # If nothing is found, return
    if len(results) == 0:
        print("No results found for {}".format(subject_id))
        return

    # look for duplicate entries, taking the last by processing time
    results["session_name"] = [x.split("_processed")[0] for x in results["name"]]
    results = results.sort_values(by="name")
    results_no_duplicates = results.drop_duplicates(subset="session_name", keep="last")

    # If there were duplicates, make a warning and print the duplicates
    if len(results) != len(results_no_duplicates):
        duplicated = results[results.duplicated(subset="session_name", keep=False)]
        warnings.warn("Duplicate session entries in docDB")
        for index, row in duplicated.iterrows():
            print("duplicated: {}".format(row["name"]))

    return results_no_duplicates


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
    example: attach_data(da_data['processed_CO_dataID'].to_list())
    """
    if len(data_asset_IDs) > 100:
        warnings.warn("list of data_asset_IDs are way too long! likely will crash CO. ")
        return
    data_assets = generate_data_asset_attach_params(data_asset_IDs, mount_point=None)
    token = os.getenv(token_name)
    if not token:
        warnings.warn("no token created. please create token")
        return
    client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=token)
    capsule_id = os.getenv("CO_CAPSULE_ID")
    results = client.capsules.attach_data_assets(
        capsule_id=capsule_id,
        attach_params=data_assets,
    )
    return results


def get_all_df_for_nwb(filename_sessions, loc="../scratch/", interested_channels=None):
    """
    get_all_df_for_nwb gets all the dataframes for the NWB and saves it
    iteratively onto a location in '../scratch/' or a specified location
    example: get_all_df_for_nwb(filename_sessions, loc = '../scratch/kenta_dec2/'
                    , interested_channels = interested_channels)
    """
    if not os.path.exists(loc):
        os.makedirs(loc)
        print(f"Directory created: {loc}")
    else:
        print(f"Saving at: {loc}")

    print(f"Saving channels: {interested_channels}")

    df_trials = pd.DataFrame()

    for idx, nwb_file in enumerate(filename_sessions):
        nwb = nwb_utils.load_nwb_from_filename(nwb_file)
        ses_idx = nwb_file.split("/")[-1].replace("behavior_", "").rsplit("_", 1)[0]
        print(f"CURRENTLY RUNNING {idx+1}/{len(filename_sessions)}: {ses_idx}")
        print(
            "--------------------------------------------------\
            ---------------------------------------"
        )

        # sessions
        df_session = nwb_utils.create_df_session(nwb)
        df_session["ses_idx"] = ses_idx

        # trials
        df_ses_trials = nwb_utils.create_df_trials(nwb)
        df_ses_trials["ses_idx"] = ses_idx
        df_trials = pd.concat([df_trials, df_ses_trials], axis=0)

        # FIP
        df_ses_fip = nwb_utils.create_fib_df(nwb, tidy=True)
        if interested_channels:
            df_ses_fip = df_ses_fip[df_ses_fip["event"].isin(interested_channels)]

        # events
        df_ses_events = nwb_utils.create_events_df(nwb)
        df_ses_events["ses_idx"] = ses_idx

        if idx == 0:
            df_session.to_csv(loc + "df_sess.csv", index=False)
            df_ses_fip.to_csv(loc + "df_fip.csv", index=False)
            df_ses_events.to_csv(loc + "df_events.csv", index=False)
        else:
            df_session.to_csv(loc + "df_sess.csv", mode="a", index=False, header=False)

            df_ses_fip.to_csv(loc + "df_fip.csv", mode="a", index=False, header=False)
            df_ses_events.to_csv(loc + "df_events.csv", mode="a", index=False, header=False)

        # df_trials saved separately because older data have lickspout_y
        # and newer ones have lickspout_y1,y2
        # correct fix is lickspout_y == lickspout_y1 == lickspout_y2
        # and always have lickspout_y1,y2.
        # I will fix this at a later date.... code to fix this is below
        df_trials = df_trials.reset_index()
        df_trials.to_csv(loc + "df_trials.csv", index=False)


def check_avail_model_by_nwb_name(nwb_name):
    """
    check_avail_model_by_nwb_name checks all available models fitted to a given session
    nwb_name
    """
    filter = {
        "nwb_name": nwb_name,  # Session id,
    }
    projection = {
        "analysis_results.fit_settings.agent_alias": 1,
        "_id": 0,
    }
    response = requests.get(
        URL, params={"filter": json.dumps(filter), "projection": json.dumps(projection)}
    )

    if not response.json():
        # small subset of sessions need "behavior" prefix.
        filter_try2 = {
            "nwb_name": "behavior_" + nwb_name,  # Session id,
        }
        response = requests.get(
            URL,
            params={"filter": json.dumps(filter_try2), "projection": json.dumps(projection)},
        )

    fitted_models = [
        item["analysis_results"]["fit_settings"]["agent_alias"] for item in response.json()
    ]
    print(fitted_models)


def get_foraging_model_info(
    df_trials, df_sess, nwb_names, loc=None, model_name="QLearning_L2F1_CK1_softmax"
):
    """
    get_foraging_model_info: retrieves fitted foraging_model information
    df_trials: dataframe for trials (1 row per trials) from nwb_utils.create df_trials.
               saved df_trials_fm will have L_prob, R_prob, L_value, R_value,
               (if choice kernel in model), L_kernel, R_kernel
    df_sess: dataframe for sessions (1 row per session) from nwb_utils.create_df_sessions
               saved df_sess_fm will have parameters fitted for each mouse.
    nwb_names: the filenames for the nwbs, formatted
                `<SUBJECT_ID>_<SESS_YEAR-SES_MON-SES_DATE>_<SESS_TIME>.nwb`
    loc: location to save the updated df_trials, df_sess with suffix `_fm.csv`.
                If given, we will save, otherwise we will return the dataframes
    model_name: model alias to get the model information

    """
    print(f"Retrieving foraging model {model_name}")
    df_trials_fm = df_trials.copy()
    df_trials_fm = df_trials_fm.rename(columns={"animal_response": "choice"})
    df_trials_fm["choice_name"] = df_trials_fm["choice"].map({1: "right", 0: "left"})

    df_trials_fm["model_name"] = model_name
    df_trials_fm["L_prob"] = pd.NA
    df_trials_fm["R_prob"] = pd.NA
    df_trials_fm["L_value"] = pd.NA
    df_trials_fm["R_value"] = pd.NA

    # check if CK is in model_name, if so, add df_trials_fm['L_kernel]
    if "CK" in model_name:
        df_trials_fm["L_kernel"] = pd.NA
        df_trials_fm["R_kernel"] = pd.NA

    df_sess_params = []
    for nwb_name in nwb_names:
        filter = {
            "nwb_name": nwb_name,  # Session id,
            "analysis_results.fit_settings.agent_alias": model_name,
        }

        projection = {
            "analysis_results.params": 1,
            "analysis_results.fitted_latent_variables": 1,
            "_id": 0,
        }
        response = requests.get(
            URL, params={"filter": json.dumps(filter), "projection": json.dumps(projection)}
        )

        if not response.json():
            # small subset of sessions need "behavior" prefix.
            filter_try2 = {
                "nwb_name": "behavior_" + nwb_name,  # Session id,
                "analysis_results.fit_settings.agent_alias": model_name,
            }
            response = requests.get(
                URL,
                params={"filter": json.dumps(filter_try2), "projection": json.dumps(projection)},
            )
            if not response.json():
                print(f"!!!!!!!! NO modeling info for {nwb_name}")
                continue
        print(nwb_name)
        record_dict = response.json()[0]
        # Fitted parameters
        params = record_dict["analysis_results"]["params"]

        # Fitted latent variables
        fitted_latent = record_dict["analysis_results"]["fitted_latent_variables"]

        # pull the information
        ses_idx = "_".join(nwb_name.split("_")[:2])
        num_trials = df_sess.loc[df_sess["ses_idx"] == ses_idx, "total_trials"].values[
            0
        ]  # all trials, 0, 1, 2
        qvals, choice_kernel, choice_prob = (np.full((2, num_trials), np.nan) for _ in range(3))
        mouse_choice_idx = df_trials_fm.index[
            (df_trials_fm["ses_idx"] == ses_idx) & (df_trials_fm["choice"] < 2)
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

        params["ses_idx"] = ses_idx
        df_sess_params.append(params)

    df_sess_params = pd.DataFrame(df_sess_params)
    df_sess_fm = df_sess.merge(df_sess_params, how="left", on=["ses_idx"])

    if loc:
        df_sess_fm.to_csv(loc + "df_sess_fm.csv")
        df_trials_fm.to_csv(loc + "df_trials_fm.csv")

    else:
        return df_trials_fm, df_sess_fm
