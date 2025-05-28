"""
Important utility functions for enriching the dataframes

    enrich_df_trials_fm
    get_df_fip_trials
    remove_tonic_df_fip
"""

import numpy as np
import pandas as pd
import itertools


def enrich_df_trials_fm(df_trials_fm):
    """
    enrich_df_trials_fm: enriches the df_trials_fm with additional columns like
                         RPE_earned, RPE_all, Q_chosen, Q_unchosen, Q_sum, Q_Delta, Q_change
                         model information can be found in repo: aind-dynamic-foraging-models
                         model fitting is done through this repo:
                         aind-foraging-behavior-bonsai-trigger-pipeline
    RPE_earned: earned_reward - chosen_values
    RPE_all: (earned_reward + extra_reward) - chosen_values
    """
    df_trials_fm_enriched = pd.DataFrame()
    models = df_trials_fm['model_name'].unique()
    sessions = np.unique(df_trials_fm['ses_idx'])
    for i_iter, (ses_idx, model_name) in enumerate(itertools.product(sessions[:], models)):
        df_ses = df_trials_fm[(df_trials_fm['ses_idx'] == ses_idx) & (df_trials_fm['model_name'] == model_name)]  # noqa: E501
        choices = df_ses.choice.map({0: 'L', 1: 'R', 2: 'I'}).values
        chosen_values = np.nan * np.zeros(len(df_ses))
        unchosen_values = np.nan * np.zeros(len(df_ses))
        chosen_kernels = np.nan * np.zeros(len(df_ses))
        unchosen_kernels = np.nan * np.zeros(len(df_ses))
        chosen_probabilities = np.nan * np.zeros(len(df_ses))
        unchosen_probabilities = np.nan * np.zeros(len(df_ses))
        chosen_stay_probabilities = np.nan * np.zeros(len(df_ses))
        # chosen_licks = np.nan * np.zeros(len(df_ses))
        for i_idx in range(len(df_ses)):
            choice = choices[i_idx]
            if choice == 'I':  # no models track ignore now
                chosen_values[i_idx] = np.nan
                chosen_kernels[i_idx] = np.nan
                chosen_probabilities[i_idx] = np.nan
                chosen_stay_probabilities[i_idx] = np.nan
                # chosen_licks[i_idx] = np.nan
            else:
                chosen_values[i_idx] = df_ses[choice+'_value'].values[i_idx]
                chosen_kernels[i_idx] = df_ses[choice+'_kernel'].values[i_idx]
                chosen_probabilities[i_idx] = df_ses[choice+'_prob'].values[i_idx]
                if choice != 'I':
                    unchosen_values[i_idx] = df_ses[{'L': 'R', 'R': 'L'}[choice]+'_value'].values[i_idx]  # noqa: E501
                    unchosen_probabilities[i_idx] = df_ses[{'L': 'R', 'R': 'L'}[choice]+'_prob'].values[i_idx]  # noqa: E501
                    unchosen_kernels[i_idx] = df_ses[{'L': 'R', 'R': 'L'}[choice]+'_kernel'].values[i_idx]  # noqa: E501
                    # chosen_licks[i_idx] = df_ses['licks '+choice].values[i_idx]
                if i_idx < len(df_ses)-1:
                    chosen_stay_probabilities[i_idx] = df_ses[choice+'_prob'].values[i_idx+1]
        for i_mod, mod in enumerate(models):
            df_ses.loc[:, 'Q_chosen'] = chosen_values
            df_ses.loc[:, 'Q_unchosen'] = unchosen_values
            df_ses.loc[:, 'Q_sum'] = df_ses['L_value'].values+df_ses['R_value'].values
            df_ses.loc[:, 'Q_Delta'] = df_ses['Q_chosen'].values-df_ses['Q_unchosen'].values
            df_ses.loc[:, 'Q_change'] = np.concatenate([[0], np.diff(chosen_values)])

            df_ses.loc[:, 'P_chosen'] = chosen_probabilities
            df_ses.loc[:, 'P_unchosen'] = unchosen_probabilities
            df_ses.loc[:, 'P_sum'] = df_ses['L_prob'].values+df_ses['R_prob'].values
            df_ses.loc[:, 'P_Delta'] = df_ses['P_chosen'].values-df_ses['P_unchosen'].values
            df_ses.loc[:, 'P_change'] = np.concatenate([[0], np.diff(chosen_probabilities)])

            df_ses.loc[:, 'K_chosen'] = chosen_kernels
            df_ses.loc[:, 'K_unchosen'] = unchosen_kernels
            df_ses.loc[:, 'K_sum'] = df_ses['L_kernel'].values+df_ses['R_kernel'].values
            df_ses.loc[:, 'K_Delta'] = df_ses['K_chosen'].values-df_ses['K_unchosen'].values
            df_ses.loc[:, 'K_change'] = np.concatenate([[0], np.diff(chosen_kernels)])

            df_ses.loc[:, 'Cprobstay'] = chosen_stay_probabilities
            df_ses.loc[:, 'RPE_earned'] = df_ses['earned_reward'].astype(float) - chosen_values
            df_ses.loc[:, 'RPE_all'] = (df_ses['earned_reward'].astype(float) +
                                        df_ses['extra_reward'].astype(float)) - chosen_values
            df_ses.loc[pd.isna(df_ses['choice']), 'RPE_earned'] = np.nan
            df_ses.loc[pd.isna(df_ses['choice']), 'RPE_all'] = np.nan
            # df_ses.loc[:,['licks_chosen']] = chosen_licks
        df_trials_fm_enriched = pd.concat([df_trials_fm_enriched, df_ses], axis=0)
    return df_trials_fm_enriched
