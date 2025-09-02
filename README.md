# aind-dynamic-foraging-data-utils


[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)



# Scope
Purpose: Ingests NWB and spits out dataframes with the relevant information. Focused on dynamic foraging. Other tasks can branch and build task-specific utils.
Inputs are nwbs, outputs are dataframes (tidy and not)
Dependencies: xarray (includes numpy and pandas), scikit-learn (includes scipy), matplotlib




# Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

# Usage

## Accessing data from an NWB file
To load an NWB file
```
import aind_dynamic_foraging_data_utils.nwb_utils as nwb_utils
nwb = nwb_utils.load_nwb_from_filename(<filepath>)
```

To extract a pandas dataframe of trials
```
df_trials = nwb_utils.create_df_trials(nwb)
```

To extract a pandas dataframe of events
```
df_events = nwb_utils.create_events_df(nwb)
```

To extract a pandas dataframe of photometry data
```
fip_df = nwb_utils.create_fib_df(nwb)
```

By default, all of these functions adjust timestamps such that t(0) is the time of the first go cue. If you wish to disable this feature, use `adjust_time=False`

## Time alignment tools
To align a data variable to a set of timepoints and create an event triggered response use the alignment module. For example to align FIP data to each go cue:

```
import aind_dynamic_foraging_data_utils.alignment as alignment

etr = alignment.event_triggered_response(
    fip_df.query('event == "<FIP channel>"'),
    "timestamps",
    "data",
    df_trials['goCue_start_time_in_session'].values,
    t_start = 0,
    t_end = 1,
    output_sampling_rate=40
    )
```


## Code ocean utility code

To attach data, you'll want to [create a token on code ocean ](https://docs.codeocean.com/user-guide/code-ocean-api/authentication#to-create-an-access-token) with all read/write permissions. Make sure to attach your token on your capsule. 

Then, you should be able to access the token via `os.getenv(token_name)`. 

### Get list of assets
To get a list of code ocean assets for a subject
```
import aind_dynamic_foraging_data_utils.code_ocean_utils as cou
results = cou.get_subject_assets(my_id)
```

Users can give a list of required data modalities
```
import aind_dynamic_foraging_data_utils.code_ocean_utils as co
# FIP data
results = co.get_subject_assets(<subject_id>, modality=['fib'])

# FIP and behavior-videos
results = co.get_subject_assets(<subject_id>, modality=['fib','behavior-videos'])

# any modalities (default)
results = co.get_subject_assets(<subject_id>, modality=[])
```

Or supply an additional filter string
```
results = co.get_subject_assets(<subject_id>, extra_filter = <my docdb query string>)
```

Or filter by a task type:
```
results = co.get_subject_assets(<subject_id>, task=['Uncoupled Baiting', 'Coupled Baiting'])
```

### Attach data
The 'code_ocean_asset_id' column gives you the data asset ID's on Code Ocean. the 'id' column is the docDB id.  

To attach a long list of data, simply call 


```
cou.attach_data(results['code_ocean_asset_id'].values)
results = co.add_data_asset_path(results)
```

with results as the dataframe from 'get_subject_assets', and 'code_ocean_asset_id' the 16 digit data asset ID from code ocean. 

### Load data
To get the dataframes from the NWBs, you can call function 

```
filename_sessions = glob.glob(f"../data/**/nwb/behavior**")
SAVED_LOC = '../scratch/dfs'
interested_channels = ['G_1_dff-poly', 'R_1_dff-poly', 'R_2_dff-poly']

get_all_df_for_nwb(filename_sessions, loc = SAVED_LOC, interested_channels = interested_channels)
```

where filename_sessions are the folder locations for the nwbs, loc is a folder location where the dataframes will be saved, interested channels are the channels you want to save for df_fip. 

All dataframes are saved per session, other than df_trials (this is because some df_trials have 2 y coordinates for the lick tube, some have 1). 

To load the dataframes, use: 

```
df_sess = pd.read_csv(SAVED_LOC + 'df_sess.csv', index_col = False)
df_events = pd.read_csv(SAVED_LOC + 'df_events.csv', index_col = False)
df_trials = pd.read_csv(SAVED_LOC + 'df_trials.csv', index_col = 0)
df_fip = pd.read_csv(SAVED_LOC + 'df_fip.csv', index_col = False)

```

To check what available fitted models we already have for each session, you can check with: 

```
check_avail_model_by_nwb_name('746345_2024-11-22_09-55-54.nwb')
```

where you input the name of the session (formatted as `<subject_ID>_<collection_date>_<collection_time>.nwb`; sometimes a prefix of `behavior_` is needed). Currently the models that are fitted on all sessions should include: 

['QLearning_L2F1_softmax', 'QLearning_L1F1_CK1_softmax', 'WSLS', 'QLearning_L1F0_epsi', 'QLearning_L2F1_CK1_softmax']

You can find out more about these models by [going here. ]([url](https://foraging-behavior-browser.allenneuraldynamics-test.org/RL_model_playground#all-available-foragers))


To enrich `df_sessions` and `df_trials` with the model information, you can use

```
nwb_name_for_models = [filename.split('/')[-1].replace('behavior_', '') for filename in filename_sessions]
SAVED_LOC = '../scratch/dfs'
get_foraging_model_info(df_trials, df_sess, nwb_name_for_models, loc = SAVED_LOC)
```

df_trials and df_sess are dataframes created from `get_all_df_for_nwb` and `nwb_name_for_models` formatted the same way for `check_avail_model_by_nwb_name`. 


# Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
