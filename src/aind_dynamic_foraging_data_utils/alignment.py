"""
    Tools for aligning continuous and discrete events
    functions:
    get_time_array
    slice_inds_and_offsets
    index_of_nearest_value
    event_triggered_response
"""

import numpy as np
import pandas as pd


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


def event_triggered_response(  # noqa C901
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
    censor=True,
    censor_times=None,
    nan_policy="error",
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
    censor: Boolean
        if True (default), censor observations that take place after the next event time
        if False, do not censor
    censor_times: list or array or None
        if None, and censor is True, then use event_times as the censor times
        if times are provided, then these are the times at which ETR is censored
    nan_policy: How to handle NaNs in the input data
        "error": raise an exception if NaNs are present in the time window of an ETR
        "interpolate": interpolate over NaN values
        "exclude": exclude any response with a NaN in the response window

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

    assert (not censor) or (output_format == "tidy"), "cannot censor data in wide output"

    assert nan_policy in ["error", "interpolate", "exclude"], "unrecognized nan_policy"

    if censor:
        event_times = np.sort(event_times)

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

        # Add one extra timestep in the data outside the t_window
        # this ensures we are not extrapolating at the edge of t_window
        # and instead doing interpolation.
        dt = np.diff(data[t].values).mean()

        # iterate over all event times
        data_reindexed = data.set_index(t, inplace=False)

        for event_number, event_time in enumerate(np.array(event_times)):
            # get a slice of the input data surrounding each event time
            data_slice = data_reindexed[y].loc[
                event_time + t_start - dt : event_time + t_end + dt
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

            elif np.any(np.isnan(data_slice)):
                if nan_policy == "error":
                    # raise exception
                    raise Exception("NaN value in data slice, at event time {}".format(event_time))
                elif nan_policy == "exclude":
                    # exclude this event
                    data_dict.update(
                        {
                            "event_{}_t={}".format(event_number, event_time): np.full(
                                len(t_array), np.nan
                            )
                        }
                    )
                else:
                    # Interpolate over NaNs
                    x_data = data_slice[~data_slice.isnull()]
                    data_slice[:] = np.interp(
                        data_slice.index.values, x_data.index.values, x_data.values
                    )

                    interpolated = np.interp(
                        data_dict["time"],
                        data_slice.index - event_time,
                        data_slice.values,
                    )
                    # Screen for extrapolation points
                    interpolated[data_dict["time"] < np.min(data_slice.index - event_time)] = np.nan
                    interpolated[data_dict["time"] > np.max(data_slice.index - event_time)] = np.nan
                    data_dict.update(
                        {"event_{}_t={}".format(event_number, event_time): interpolated}
                    )
            else:
                # update our dictionary to have a new key defined as
                # 'event_{EVENT NUMBER}_t={EVENT TIME}' and
                # a value that includes an array that represents the
                # sliced data around the current event, interpolated
                # on the linearly spaced time array
                interpolated = np.interp(
                    data_dict["time"],
                    data_slice.index - event_time,
                    data_slice.values,
                )
                # Screen for extrapolation points
                interpolated[data_dict["time"] < np.min(data_slice.index - event_time)] = np.nan
                interpolated[data_dict["time"] > np.max(data_slice.index - event_time)] = np.nan
                data_dict.update({"event_{}_t={}".format(event_number, event_time): interpolated})

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
    if censor:
        tidy_etr = censor_event_triggered_response(
            tidy_etr, y, t_start, t_end, event_times, censor_times
        )
    return tidy_etr


def censor_event_triggered_response(etr, y, t_start, t_end, event_times, censor_times=None):
    """
    censors the event triggered response by the immediately preceeding or
    subsequent event times if that event time is within the (t_start, t_end)
    time window

    censored timepoints are replaced with NaN, so all data points are still present

    etr: dataframe, event triggered response
    y: column of the response variable to censor
    t_start: start of event triggered response window
    t_end: end of event triggered response window
    censor: Boolean
        if True, censor observations that take place after the next event time
        if False, do not censor
    censor_times: list or array or None
        if None, and censor is True, then use event_times as the censor times
        if times are provided, then these are the times at which ETR is censored
    """

    if censor_times is None:
        # Compute when we should censor
        diff = np.diff(event_times)
        diff_backward = np.concatenate([[np.inf], diff])
        diff_forward = np.concatenate([diff, [np.inf]])
        backward_time = [-np.min([np.abs(t_start), x]) for x in diff_backward]
        forward_time = [np.min([t_end, x]) for x in diff_forward]

        # double check we have all events
        assert len(event_times) == len(etr["event_number"].unique()), "event times missing"

        # Censor trials
        for index, time in enumerate(event_times):
            vec = (etr["event_number"] == index) & (etr["time"] < backward_time[index])
            etr.loc[vec, y] = np.nan
            vec = (etr["event_number"] == index) & (etr["time"] > forward_time[index])
            etr.loc[vec, y] = np.nan

        return etr
    else:
        censor_times = np.sort(censor_times)
        backward_time = []
        forward_time = []
        for e in event_times:
            before = censor_times[censor_times < e]
            if len(before) == 0:
                backward_time.append(t_start)
            else:
                backward_time.append(np.max([t_start, before[-1] - e]))
            after = censor_times[censor_times > e]
            if len(after) == 0:
                forward_time.append(t_end)
            else:
                forward_time.append(np.min([t_end, after[0] - e]))

        # Censor trials
        for index, time in enumerate(event_times):
            vec = (etr["event_number"] == index) & (etr["time"] < backward_time[index])
            etr.loc[vec, y] = np.nan
            vec = (etr["event_number"] == index) & (etr["time"] > forward_time[index])
            etr.loc[vec, y] = np.nan

        return etr
