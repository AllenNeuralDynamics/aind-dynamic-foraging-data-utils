"""
Tests for the DuckDB query helpers (foraging_cache.query).

Builds a tiny local cache from the real NWB files in tests/nwb/, writes a minimal
session_table.parquet whose keys match the built trials, then exercises the helpers
against that local build via ``base=`` — no S3 / network needed.
"""

import glob
import os
import tempfile
import unittest

import duckdb
import pandas as pd

from aind_dynamic_foraging_data_utils.foraging_cache import query
from aind_dynamic_foraging_data_utils.foraging_cache.util import parquet_builder

TEST_NWB_DIR = "./tests/nwb"


def _build_local_cache(tmpdir):
    """Build trial+event tables from tests/nwb/ and a matching session_table.parquet.

    Returns (session_path, trial_prefix, event_prefix). The session table's ``_session_id``
    is taken from the built trials so the helper joins line up exactly, plus two fake
    metadata columns (``task``, ``foraging_eff``) to exercise filtering + metadata join.
    """
    nwb_files = glob.glob(os.path.join(TEST_NWB_DIR, "*.nwb"))
    rows = []
    for fpath in nwb_files:
        parsed = parquet_builder._parse_nwb_filename(fpath)
        if parsed is None:
            continue
        subject_id, session_date, nwb_suffix = parsed
        rows.append({"subject_id": subject_id, "session_date": session_date,
                     "nwb_suffix": nwb_suffix, "data_source": "bonsai",
                     "co_asset_id": None, "co_s3_nwb_uri": None,
                     "nwb_data_source": "bonsai_s3"})
    session_df = pd.DataFrame(rows)

    trial_prefix = os.path.join(tmpdir, "trial_table")
    event_prefix = os.path.join(tmpdir, "event_table")
    nwb_index = parquet_builder.build_nwb_file_index(bonsai_dir=TEST_NWB_DIR, bpod_dir=TEST_NWB_DIR)
    parquet_builder.build_trial_and_event_tables(
        session_df=session_df, trial_output_prefix=trial_prefix,
        event_output_prefix=event_prefix, nwb_file_index=nwb_index,
        incremental=False, verbose=False)

    # Derive a session table from the built trials so _session_id matches session_id.
    sess = duckdb.sql(
        f"SELECT DISTINCT session_id AS _session_id, CAST(subject_id AS VARCHAR) AS subject_id, "
        f"session_date FROM read_parquet('{trial_prefix}/**/*.parquet', "
        f"hive_partitioning=true, union_by_name=true)").df()
    sess["task"] = "TestTask"
    sess["foraging_eff"] = 0.5
    session_path = os.path.join(tmpdir, "session_table.parquet")
    sess.to_parquet(session_path, index=False)
    return session_path, trial_prefix, event_prefix


class TestQueryHelpers(unittest.TestCase):
    """End-to-end helper tests against a local build."""

    def test_select_fetch_and_escape_hatch(self):
        """select_sessions -> fetch_trials/fetch_events + the read_trials escape hatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path, trial_prefix, event_prefix = _build_local_cache(tmpdir)

            # --- select_sessions: metric filter, carrying metadata ---
            sel = query.select_sessions("foraging_eff > 0", base=session_path,
                                        columns=["task", "foraging_eff"])
            self.assertGreater(len(sel), 0)
            self.assertIn("_session_id", sel.columns)
            self.assertEqual(list(sel.columns[:3]), ["_session_id", "subject_id", "session_date"])

            # --- fetch_trials: scoped read + metadata join ---
            tr = query.fetch_trials(sel, base=trial_prefix,
                                    columns=["animal_response", "earned_reward"])
            self.assertGreater(len(tr), 0)
            self.assertEqual(list(tr.columns[:3]), ["subject_id", "session_date", "session_id"])
            for col in ["task", "foraging_eff", "animal_response", "earned_reward"]:
                self.assertIn(col, tr.columns)
            # exactly the selected sessions, nothing extra
            self.assertTrue(set(tr["session_id"]).issubset(set(sel["_session_id"])))
            # ordered by subject_id, session_date, trial
            self.assertTrue(tr["subject_id"].is_monotonic_increasing)

            # --- fetch_events: with event-type filter ---
            ev = query.fetch_events(sel, base=event_prefix,
                                    events=["left_lick_time", "right_lick_time"],
                                    columns=["trial", "timestamps", "event"])
            self.assertGreater(len(ev), 0)
            self.assertEqual(set(ev["event"].unique()), {"left_lick_time", "right_lick_time"})

            # --- read_trials escape hatch: scoped source drives arbitrary SQL ---
            subs = sel["subject_id"].unique().tolist()
            src = query.read_trials(subs, base=trial_prefix)
            n_scoped = duckdb.sql(f"SELECT COUNT(*) c FROM {src}").df().c[0]
            n_all = duckdb.sql(
                f"SELECT COUNT(*) c FROM read_parquet('{trial_prefix}/**/*.parquet', "
                f"hive_partitioning=true, union_by_name=true)").df().c[0]
            self.assertEqual(n_scoped, n_all)  # only subject(s) in this tiny cache

    def test_read_trials_edge_cases(self):
        """read_trials: full-glob fallback and empty-source path for a missing subject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, trial_prefix, _ = _build_local_cache(tmpdir)
            # subjects=None -> full glob
            self.assertIn("**", query.read_trials(base=trial_prefix))
            # a subject with no partition -> safe empty source (no error)
            empty_src = query.read_trials(["000000"], base=trial_prefix)
            self.assertEqual(int(duckdb.sql(f"SELECT COUNT(*) c FROM {empty_src}").df().c[0]), 0)

    def test_fetch_empty_selection(self):
        """fetch_trials returns an empty frame when the session selection is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path, trial_prefix, _ = _build_local_cache(tmpdir)
            empty = query.select_sessions("foraging_eff > 999", base=session_path)
            self.assertEqual(len(empty), 0)
            self.assertEqual(len(query.fetch_trials(empty, base=trial_prefix)), 0)


if __name__ == "__main__":
    unittest.main()
