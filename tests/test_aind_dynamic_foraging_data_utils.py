"""Testing for utility functions for processing dynamic foraging data."""

import unittest

import numpy as np
import pandas as pd
import glob

from src.aind_dynamic_foraging_data_utils import alignment, nwb_utils


class DynamicForagingTest(unittest.TestCase):
    """
    A class for testing the dynamic_foraging_utils module.
    """

    def test_create_df_session(self):
        """
        tests the `create_df_session` function
        """
        nwb_files = glob.glob("./tests/nwb/**.nwb")
        # create a dataframe for each nwb file

        df = nwb_utils.create_df_session(nwb_files)
        assert len(df) == len(nwb_files)

        df = nwb_utils.create_df_session(nwb_files[0])
        assert len(df) == 1

    def test_create_df_trials(self):
        """
        tests the `create_df_trials` function
        """
        nwb_files = glob.glob("./tests/nwb/**.nwb")
        # create a dataframe for one nwb file
        df = nwb_utils.create_df_trials(nwb_files[0])
        # check that the dataframe has correct session names
        session_name = "_".join(nwb_files[0].split("/")[-1].split("_")[0:-1])
        assert df.ses_idx[0] == session_name, "{} != {}".format(session_name, df.ses_idx[0])

    def test_get_time_array_with_sampling_rate(self):
        """
        tests the `get_time_array` function while passing
        a sampling_rate argument
        """
        # this should give [-1, 1) with steps of 0.5, exclusive of the endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=1, sampling_rate=2, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=1, sampling_rate=2, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5, 1.0])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly divisible by the
        # step size (0.5), the array should end before the desired endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=0.75, sampling_rate=2, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly divisible by the
        # step size (0.5), the array should end before the desired endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=0.75, sampling_rate=2, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

    def test_get_time_array_with_step_size(self):
        """
        tests the `get_time_array` function while passing a step_size argument
        """
        # this should give [-1, 1) with steps of 0.5, exclusive of the endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=1, step_size=0.5, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=1, step_size=0.5, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5, 1.0])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly
        # divisible by the step size (0.5), the array should
        # end before the desired endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=0.75, step_size=0.5, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly
        # divisible by the step size (0.5),
        # the array should end before the desired endpoint
        t_array = alignment.get_time_array(
            t_start=-1, t_end=0.75, step_size=0.5, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

    def test_event_triggered_response(self):
        """
        tests the `test_event_triggered_response` function
        """
        # make a time vector from -10 to 110
        t = np.arange(-10, 110, 0.01)

        # Make a dataframe with one column as time, and another
        # column called 'sinusoid' defined as sin(2*pi*t)
        # The sinusoid column will have a period of 1
        df = pd.DataFrame({"time": t, "sinusoid": np.sin(2 * np.pi * t)})
        df_copy = df.copy(deep=True)

        # Make an event triggered response
        etr = alignment.event_triggered_response(
            data=df,
            t="time",
            y="sinusoid",
            event_times=np.arange(100),
            t_before=1,
            t_after=1,
            output_sampling_rate=100,
        )

        # Assert that the average value of the agrees with expectations
        assert np.isclose(etr.query("time == 0")["sinusoid"].mean(), 0, rtol=0.01)
        assert np.isclose(etr.query("time == 0.25")["sinusoid"].mean(), 1, rtol=0.01)
        assert np.isclose(etr.query("time == 0.5")["sinusoid"].mean(), 0, rtol=0.01)
        assert np.isclose(etr.query("time == 0.75")["sinusoid"].mean(), -1, rtol=0.01)
        assert np.isclose(etr.query("time == 1")["sinusoid"].mean(), 0, rtol=0.01)

        # Assert that the dataframe is unchanged
        pd.testing.assert_frame_equal(df, df_copy)

    def test_event_triggered_response_censor(self):
        """
        tests the censoring property of the event_triggered_response function
        """
        # make a sample test set
        t = np.arange(0, 10, 0.01)
        y = np.array([10] * len(t))
        event_times = [2, 3, 4, 5, 6, 7, 8]
        for e in event_times:
            y[(t > e) & (t < (e + 0.25))] = 1

        df = pd.DataFrame({"time": t, "y": y})

        # make etr
        etr_censored = alignment.event_triggered_response(
            data=df,
            t="time",
            y="y",
            event_times=event_times,
            t_before=2,
            t_after=2,
            output_sampling_rate=100,
            censor=True,
        )

        # assert properties of etr
        assert np.isclose(etr_censored.query("time > 1")["y"].mean(), 10, rtol=0.01)
        assert np.isclose(etr_censored.query("time < -1")["y"].mean(), 10, rtol=0.01)

        # Test with arbitrary list of censor times
        censor_times = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
        etr_censored = alignment.event_triggered_response(
            data=df,
            t="time",
            y="y",
            event_times=event_times,
            t_before=2,
            t_after=2,
            output_sampling_rate=100,
            censor=True,
            censor_times=censor_times,
        )

        # assert properties of etr
        assert np.isclose(etr_censored.query("time > 1")["y"].mean(), 10, rtol=0.01)
        assert np.isclose(etr_censored.query("time < 0")["y"].mean(), 10, rtol=0.01)

    def test_event_triggered_response_nan_policy(self):
        """
        tests the `test_event_triggered_response` function
        """
        # make a time vector from -10 to 110
        t = np.arange(-10, 110, 0.01)

        # Make a dataframe with one column as time, and another
        # column called 'sinusoid' defined as sin(2*pi*t)
        # The sinusoid column will have a period of 1
        df = pd.DataFrame({"time": t, "sinusoid": np.sin(2 * np.pi * t)})

        # Make an event triggered response, NaN values are outside window
        df.loc[0:100, "sinusoid"] = np.nan
        etr = alignment.event_triggered_response(
            data=df,
            t="time",
            y="sinusoid",
            event_times=np.arange(100),
            t_before=1,
            t_after=1,
            output_sampling_rate=100,
            nan_policy="error",
        )

        # Raises an error
        df.loc[3980:4050, "sinusoid"] = np.nan
        with self.assertRaises(Exception):
            alignment.event_triggered_response(
                data=df,
                t="time",
                y="sinusoid",
                event_times=np.arange(100),
                t_before=1,
                t_after=1,
                output_sampling_rate=100,
                nan_policy="error",
            )

        # outputs are NaNs around NaN data
        etr = alignment.event_triggered_response(
            data=df,
            t="time",
            y="sinusoid",
            event_times=np.arange(100),
            t_before=1,
            t_after=1,
            output_sampling_rate=100,
            nan_policy="exclude",
        )
        assert np.isclose(
            etr.query("event_number == 29")["sinusoid"].isnull().mean(), 1.0, rtol=0.01
        )
        assert np.isclose(
            etr.query("event_number == 30")["sinusoid"].isnull().mean(), 1.0, rtol=0.01
        )
        assert np.isclose(
            etr.query("event_number == 31")["sinusoid"].isnull().mean(), 1.0, rtol=0.01
        )
        assert np.isclose(
            etr.query("event_number not in [29,30,31]")["sinusoid"].isnull().mean(), 0.0, rtol=0.01
        )

        # outputs are NaNs around NaN data
        etr = alignment.event_triggered_response(
            data=df,
            t="time",
            y="sinusoid",
            event_times=np.arange(100),
            t_before=1,
            t_after=1,
            output_sampling_rate=100,
            nan_policy="interpolate",
        )
        assert np.isclose(etr["sinusoid"].isnull().mean(), 0.0, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
