"""
    This is an example script to extract dataframes
    from an NWB file
"""

import glob

from aind_dynamic_foraging_data_utils import nwb_utils

# List of filepaths to NWBs
NWB_FILES = glob.glob("../tests/nwb/**.nwb")

# Load a single NWB file
nwb = nwb_utils.load_nwb_from_filename(NWB_FILES[0])

# Make dataframe of sessions, each row is a session
# Accepts: NWB objects, NWB files, list of NWB files
df_session = nwb_utils.create_df_session(NWB_FILES)

# Dataframe of trials, each row is a trial
# Accepts NWB filepath or object
df_trials = nwb_utils.create_df_trials(nwb)
df_trials = nwb_utils.create_df_trials(NWB_FILES[0])

# tidy data frame of all events in the session
# Accepts NWB filepath or object
df_events = nwb_utils.create_events_df(nwb)
df_events = nwb_utils.create_events_df(NWB_FILES[0])

# Data frame of FIB data
# Accepts NWB filepath or object
fib_df = nwb_utils.create_fib_df(nwb)
fib_df = nwb_utils.create_fib_df(NWB_FILES[0])
