"""
Tests for parquet_builder and cache_utils modules.

Test structure:
  1. Unit tests for _parse_nwb_filename (no I/O)
  2. Unit tests for build_nwb_file_index (uses a temp directory)
  3. Integration test: build_session_table → parquet round-trip (mocked Han session table)
  4. Integration test: build_trial_and_event_tables → cache_utils query round-trip
     (uses real NWB files from tests/nwb/)
"""

import os
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

from aind_dynamic_foraging_data_utils import cache_utils
from aind_dynamic_foraging_data_utils.foraging_cache import parquet_builder


# ---------------------------------------------------------------------------
# 1. _parse_nwb_filename
# ---------------------------------------------------------------------------


class TestParseNwbFilename(unittest.TestCase):
    """Unit tests for _parse_nwb_filename."""

    def test_old_format(self):
        """Old bonsai / bpod format: {subject_id}_{date}_{HH-MM-SS}.nwb"""
        result = parquet_builder._parse_nwb_filename("688237_2023-08-11_11-47-36.nwb")
        self.assertEqual(result, ("688237", "2023-08-11", 114736))

    def test_new_behavior_format(self):
        """New bonsai format: behavior_{subject_id}_{date}_{HH-MM-SS}.nwb"""
        result = parquet_builder._parse_nwb_filename("behavior_771432_2024-12-16_13-06-12.nwb")
        self.assertEqual(result, ("771432", "2024-12-16", 130612))

    def test_absolute_path_new_format(self):
        """Full absolute path should still parse correctly."""
        result = parquet_builder._parse_nwb_filename(
            "/data/foraging_nwb_bonsai/behavior_123456_2024-01-01_09-00-00.nwb"
        )
        self.assertEqual(result, ("123456", "2024-01-01", 90000))

    def test_non_nwb_file_returns_none(self):
        """Non-.nwb files should return None."""
        self.assertIsNone(parquet_builder._parse_nwb_filename("bonsai_pipeline.log"))
        self.assertIsNone(parquet_builder._parse_nwb_filename("mice_pi_mapping.json"))

    def test_malformed_name_returns_none(self):
        """Names that don't match either pattern should return None."""
        self.assertIsNone(parquet_builder._parse_nwb_filename("unknown_file.nwb"))
        self.assertIsNone(parquet_builder._parse_nwb_filename("no_date.nwb"))

    def test_nwb_suffix_is_int(self):
        """nwb_suffix must be an int, not a string."""
        result = parquet_builder._parse_nwb_filename("111111_2024-11-18_14-10-48.nwb")
        self.assertIsInstance(result[2], int)
        self.assertEqual(result[2], 141048)


# ---------------------------------------------------------------------------
# 2. build_nwb_file_index
# ---------------------------------------------------------------------------


class TestBuildNwbFileIndex(unittest.TestCase):
    """Unit tests for build_nwb_file_index using a temporary directory."""

    def setUp(self):
        """Create temporary bonsai and bpod directories with fake NWB filenames."""
        self.tmpdir = tempfile.mkdtemp()
        self.bonsai_dir = os.path.join(self.tmpdir, "bonsai")
        self.bpod_dir = os.path.join(self.tmpdir, "bpod")
        os.makedirs(self.bonsai_dir)
        os.makedirs(self.bpod_dir)

        # Create dummy files (content doesn't matter — only filenames are parsed)
        for fname in [
            "behavior_123456_2024-01-01_09-00-00.nwb",  # new bonsai format
            "789012_2023-06-15_14-30-00.nwb",            # old bonsai format
        ]:
            open(os.path.join(self.bonsai_dir, fname), "w").close()

        for fname in [
            "789012_2023-06-15_14-30-00.nwb",  # same session as bonsai — bonsai should win
            "555555_2022-03-10_08-00-00.nwb",  # bpod-only session
        ]:
            open(os.path.join(self.bpod_dir, fname), "w").close()

    def test_index_size(self):
        """Index should have 3 unique sessions (bonsai overwrites bpod on collision)."""
        index = parquet_builder.build_nwb_file_index(self.bonsai_dir, self.bpod_dir)
        self.assertEqual(len(index), 3)

    def test_bonsai_priority_over_bpod(self):
        """When same session exists in both dirs, bonsai path should be in index."""
        index = parquet_builder.build_nwb_file_index(self.bonsai_dir, self.bpod_dir)
        key = ("789012", "2023-06-15", 143000)
        self.assertIn(key, index)
        self.assertTrue(index[key].startswith(self.bonsai_dir))

    def test_bpod_only_session_present(self):
        """Sessions only in bpod dir should still be indexed."""
        index = parquet_builder.build_nwb_file_index(self.bonsai_dir, self.bpod_dir)
        key = ("555555", "2022-03-10", 80000)
        self.assertIn(key, index)
        self.assertTrue(index[key].startswith(self.bpod_dir))

    def test_keys_are_tuples(self):
        """All keys must be (subject_id_str, session_date_str, nwb_suffix_int) tuples."""
        index = parquet_builder.build_nwb_file_index(self.bonsai_dir, self.bpod_dir)
        for key in index.keys():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 3)
            self.assertIsInstance(key[0], str)   # subject_id
            self.assertIsInstance(key[1], str)   # session_date
            self.assertIsInstance(key[2], int)   # nwb_suffix


# ---------------------------------------------------------------------------
# 3. build_session_table round-trip (mocked session data)
# ---------------------------------------------------------------------------


class TestBuildSessionTableRoundTrip(unittest.TestCase):
    """
    Test build_session_table with a minimal mocked session DataFrame,
    then read it back and verify structure and content.
    """

    def _make_mock_session_df(self):
        """Return a minimal DataFrame mimicking get_session_table() output."""
        return pd.DataFrame(
            {
                "subject_id": ["633456", "633456", "712345"],
                "session_date": ["2024-01-01", "2024-01-02", "2024-03-15"],
                "nwb_suffix": [90000, 91500, 140000],
                "finished_trials": [300, 250, 400],
                "foraging_eff": [0.75, 0.80, 0.72],
                "data_source": ["bonsai", "bonsai", "bpod"],
            }
        )

    def test_round_trip_local(self):
        """Write session table to local parquet and read back; check shape and columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "session_table.parquet")

            # Build using our mock data by monkey-patching get_session_table
            import unittest.mock as mock

            mock_df = self._make_mock_session_df()
            with mock.patch(
                "aind_analysis_arch_result_access.han_pipeline.get_session_table",
                return_value=mock_df,
            ):
                df_out = parquet_builder.build_session_table(
                    output_path=out_path,
                    bowen_csv_path="/nonexistent/path.csv",  # triggers warning; returns empty set
                    include_co_assets=False,
                    verbose=False,
                )

            # Verify parquet was written and has expected shape
            self.assertTrue(os.path.exists(out_path))
            df_read = pq.read_table(out_path).to_pandas()
            self.assertEqual(len(df_read), len(mock_df))

            # Required new columns must be present
            for col in ["co_asset_id", "co_s3_nwb_uri", "is_bad_bowen_session", "nwb_data_source"]:
                self.assertIn(col, df_read.columns, f"Missing column: {col}")

    def test_nwb_data_source_assigned(self):
        """nwb_data_source should be 'bpod_s3' for bpod sessions and 'bonsai_s3' for others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "session_table.parquet")
            import unittest.mock as mock

            mock_df = self._make_mock_session_df()
            with mock.patch(
                "aind_analysis_arch_result_access.han_pipeline.get_session_table",
                return_value=mock_df,
            ):
                df_out = parquet_builder.build_session_table(
                    output_path=out_path,
                    bowen_csv_path="/nonexistent/path.csv",
                    include_co_assets=False,
                    verbose=False,
                )

            # The row with data_source="bpod" should map to "bpod_s3"
            bpod_row = df_out[df_out["data_source"] == "bpod"]
            self.assertTrue((bpod_row["nwb_data_source"] == "bpod_s3").all())

            # Bonsai rows → "bonsai_s3"
            bonsai_rows = df_out[df_out["data_source"] == "bonsai"]
            self.assertTrue((bonsai_rows["nwb_data_source"] == "bonsai_s3").all())


# ---------------------------------------------------------------------------
# 4. build_trial_and_event_tables + cache_utils round-trip
# ---------------------------------------------------------------------------


class TestTrialEventRoundTrip(unittest.TestCase):
    """
    Integration test using real NWB files from tests/nwb/.

    Builds trial and event tables for those sessions, writes them to a local temp
    directory, then reads them back via cache_utils and checks correctness.
    """

    TEST_NWB_DIR = "./tests/nwb"
    TEST_SUBJECT_ID = "697062"  # Subject present in tests/nwb/

    def _build_minimal_session_df(self, nwb_files):
        """
        Build a minimal session_df from a list of NWB file paths.
        Parses (subject_id, session_date, nwb_suffix) from filenames.
        """
        rows = []
        for fpath in nwb_files:
            parsed = parquet_builder._parse_nwb_filename(fpath)
            if parsed is None:
                continue
            subject_id, session_date, nwb_suffix = parsed
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_date": session_date,
                    "nwb_suffix": nwb_suffix,
                    "data_source": "bonsai",
                    "co_asset_id": None,
                    "co_s3_nwb_uri": None,
                    "is_bad_bowen_session": False,
                    "nwb_data_source": "bonsai_s3",
                }
            )
        return pd.DataFrame(rows)

    def test_trial_event_round_trip(self):
        """
        Build trial/event tables from test NWBs and read back via cache_utils.
        Checks row counts > 0 and required columns exist.
        """
        import glob

        nwb_files = glob.glob(os.path.join(self.TEST_NWB_DIR, "*.nwb"))
        self.assertGreater(len(nwb_files), 0, "No NWB files found in tests/nwb/")

        session_df = self._build_minimal_session_df(nwb_files)
        self.assertGreater(len(session_df), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trial_prefix = os.path.join(tmpdir, "trial_table")
            event_prefix = os.path.join(tmpdir, "event_table")
            meta_path = os.path.join(tmpdir, "build_metadata.json")

            # Build a local NWB index pointing at our test directory
            nwb_index = parquet_builder.build_nwb_file_index(
                bonsai_dir=self.TEST_NWB_DIR,
                bpod_dir=self.TEST_NWB_DIR,  # same dir; no bpod files here, that's fine
            )

            summary = parquet_builder.build_trial_and_event_tables(
                session_df=session_df,
                trial_output_prefix=trial_prefix,
                event_output_prefix=event_prefix,
                nwb_file_index=nwb_index,
                build_metadata_path=meta_path,
                incremental=False,
                verbose=False,
            )

            # At least some sessions should be processed
            self.assertGreater(summary["n_processed"], 0)

            # Read back via cache_utils
            df_trials = cache_utils.get_trial_table(
                subject_ids=[self.TEST_SUBJECT_ID],
                trial_table_prefix=trial_prefix,
            )
            df_events = cache_utils.get_event_table(
                subject_ids=[self.TEST_SUBJECT_ID],
                event_table_prefix=event_prefix,
            )

            # Both tables should be non-empty
            self.assertGreater(len(df_trials), 0, "Trial table is empty")
            self.assertGreater(len(df_events), 0, "Event table is empty")

            # Required columns in trial table
            for col in ["subject_id", "session_date", "nwb_suffix", "session_id",
                        "earned_reward", "animal_response", "nwb_data_source"]:
                self.assertIn(col, df_trials.columns, f"Trial table missing column: {col}")

            # Required columns in event table
            for col in ["subject_id", "session_id", "timestamps", "event", "nwb_data_source"]:
                self.assertIn(col, df_events.columns, f"Event table missing column: {col}")

            # All trial rows should belong to expected subject
            self.assertTrue((df_trials["subject_id"] == self.TEST_SUBJECT_ID).all())

    def test_incremental_skips_processed(self):
        """
        Second call with incremental=True should skip already-processed sessions.
        """
        import glob

        nwb_files = glob.glob(os.path.join(self.TEST_NWB_DIR, "*.nwb"))
        session_df = self._build_minimal_session_df(nwb_files)

        with tempfile.TemporaryDirectory() as tmpdir:
            trial_prefix = os.path.join(tmpdir, "trial_table")
            event_prefix = os.path.join(tmpdir, "event_table")
            meta_path = os.path.join(tmpdir, "build_metadata.json")

            nwb_index = parquet_builder.build_nwb_file_index(
                bonsai_dir=self.TEST_NWB_DIR,
                bpod_dir=self.TEST_NWB_DIR,
            )
            kwargs = dict(
                session_df=session_df,
                trial_output_prefix=trial_prefix,
                event_output_prefix=event_prefix,
                nwb_file_index=nwb_index,
                build_metadata_path=meta_path,
                verbose=False,
            )

            # First run: process everything
            summary1 = parquet_builder.build_trial_and_event_tables(**kwargs, incremental=False)

            # Second run (incremental): all sessions already processed → 0 new sessions
            summary2 = parquet_builder.build_trial_and_event_tables(**kwargs, incremental=True)

            self.assertEqual(summary1["n_failed"], 0)
            self.assertEqual(summary2["n_processed"], 0)
            self.assertEqual(summary2["n_skipped"], 0)
            # All sessions from run 1 should now be in the "already processed" set
            # so they don't appear as n_skipped either — they're pre-filtered out


if __name__ == "__main__":
    unittest.main()
