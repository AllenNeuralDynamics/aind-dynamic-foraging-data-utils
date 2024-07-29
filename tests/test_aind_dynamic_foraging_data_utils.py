"""Testing for utility functions for processing dynamic foraging data."""

import unittest

import numpy as np
import pandas as pd
import glob

from src.aind_dynamic_foraging_data_utils import aind_dynamic_foraging_data_utils


class DynamicForagingTest(unittest.TestCase):
    """
    A class for testing the dynamic_foraging_utils module.
    """
    def test_create_df_session(self):
        """
        tests the `create_df_session` function
        """
        nwb_files = glob.glob('./tests/nwb/**.nwb')
        # create a dataframe for each nwb file

        df = pd.DataFrame()
        for nwb_file in nwb_files:
            df = pd.concat([df, aind_dynamic_foraging_data_utils.create_df_session(nwb_file)])
        # check that the dataframe has the correct rows
        assert (len(df) == len(nwb_files))

    def test_create_df_trials(self):
        """
        tests the `create_df_trials` function
        """
        nwb_files = glob.glob('./tests/nwb/**.nwb')
        # create a dataframe for one nwb file
        df = aind_dynamic_foraging_data_utils.create_df_trials(nwb_files[0])
        # check that the dataframe has correct session names
        session_name = '_'.join(nwb_files[0].split('/')[-1].split('_')[:-1])
        assert (df.ses_idx[0] == session_name)

    def test_get_time_array_with_sampling_rate(self):
        """
        tests the `get_time_array` function while passing
        a sampling_rate argument
        """
        # this should give [-1, 1) with steps of 0.5, exclusive of the endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=1, sampling_rate=2, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=1, sampling_rate=2, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5, 1.0])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly divisible by the
        # step size (0.5), the array should end before the desired endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=0.75, sampling_rate=2, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly divisible by the
        # step size (0.5), the array should end before the desired endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=0.75, sampling_rate=2, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

    def test_get_time_array_with_step_size(self):
        """
        tests the `get_time_array` function while passing a step_size argument
        """
        # this should give [-1, 1) with steps of 0.5, exclusive of the endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=1, step_size=0.5, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=1, step_size=0.5, include_endpoint=True
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5, 1.0])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly
        # divisible by the step size (0.5), the array should
        # end before the desired endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
            t_start=-1, t_end=0.75, step_size=0.5, include_endpoint=False
        )
        assert (t_array == np.array([-1.0, -0.5, 0.0, 0.5])).all()

        # this should give [-1, 0.75) with steps of 0.5.
        # becuase the desired range (1.75) is not evenly
        # divisible by the step size (0.5),
        # the array should end before the desired endpoint
        t_array = aind_dynamic_foraging_data_utils.get_time_array(
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
        etr = aind_dynamic_foraging_data_utils.event_triggered_response(
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


if __name__ == "__main__":
    unittest.main()
