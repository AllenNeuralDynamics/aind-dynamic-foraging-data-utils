"""
Tests for the fast ``backend="direct"`` reader path in ``nwb_utils``.

The direct backend reads NWB data straight from the store (``h5py`` for HDF5,
``zarr.open_consolidated`` for Zarr) instead of building the full pynwb object graph.
These tests verify it returns results identical to ``backend="pynwb"`` across multiple
NWB files, and benchmark the relative speed of the two readers.

The committed sessions under ``tests/nwb`` (HDF5 + Zarr) are always exercised so CI can
validate both formats. To additionally validate/benchmark on realistic CO-asset sessions
(large, with FIP) - which is where the direct reader's advantage is largest - point the
tests at them, **and they are prioritised in the sweep**:

* ``AIND_NWB_TEST_S3``  - a JSON file of, or comma-separated, ``s3://`` NWB paths
  (e.g. produced by ``code_ocean_utils.get_assets`` + ``add_s3_location``), or
* ``AIND_NWB_TEST_DIR`` - a directory of attached ``.nwb`` sessions.
"""

import glob
import json
import os
import time
import unittest
import warnings

import pandas as pd

from src.aind_dynamic_foraging_data_utils import nwb_utils
from src.aind_dynamic_foraging_data_utils._direct_nwb import DirectNWBAdapter

# Small NWB sessions shipped with the repo: 5 HDF5 (bonsai) + 1 Zarr (AIND).
COMMITTED_FILES = sorted(glob.glob("./tests/nwb/*.nwb"))

# Cap on how many (potentially large) CO-asset sessions to pull into the sweep.
MAX_CO_ASSET_FILES = int(os.environ.get("AIND_NWB_TEST_MAX", "4"))


def discover_co_asset_nwbs(limit=MAX_CO_ASSET_FILES):
    """Return CO-asset NWB paths to prioritise (S3 first), or [] if none configured."""
    s3_spec = os.environ.get("AIND_NWB_TEST_S3")
    if s3_spec:
        if os.path.isfile(s3_spec):
            paths = json.load(open(s3_spec))
        else:
            paths = [p.strip() for p in s3_spec.split(",") if p.strip()]
        return paths[:limit]

    test_dir = os.environ.get("AIND_NWB_TEST_DIR")
    if not test_dir or not os.path.isdir(test_dir):
        return []
    found, seen = [], set()
    for pattern in ("*.nwb", "**/nwb/*.nwb", "**/*.nwb"):
        for path in sorted(glob.glob(os.path.join(test_dir, pattern), recursive=True)):
            real = os.path.realpath(path)
            if real not in seen:
                seen.add(real)
                found.append(path)
    return found[:limit]


CO_ASSET_FILES = discover_co_asset_nwbs()
# CO assets prioritised first in the correctness/benchmark sweeps.
ALL_FILES = CO_ASSET_FILES + COMMITTED_FILES

# Behaviour readers exercised on every file (cheap, no FIP).
BEHAVIOR_READERS = {
    "create_df_session": dict(),
    "create_df_trials": dict(verbose=False),
    "create_df_events": dict(verbose=False),
}


def _run(reader_name, nwb_file, backend, **kwargs):
    """Run one reader on one file with the given backend, silencing warnings."""
    func = getattr(nwb_utils, reader_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(nwb_file, backend=backend, **kwargs)


def _assert_frames_equal(result_pynwb, result_direct, msg):
    """Assert two reader outputs (DataFrame or None) are equivalent."""
    if result_pynwb is None or result_direct is None:
        assert result_pynwb is None and result_direct is None, msg
        return
    pd.testing.assert_frame_equal(
        result_pynwb.reset_index(drop=True),
        result_direct.reset_index(drop=True),
        check_dtype=False,
        obj=msg,
    )


class DirectBackendTest(unittest.TestCase):
    """Validate and benchmark the direct (h5py / zarr) reader backend."""

    def test_committed_files_present(self):
        """Both an HDF5 and a Zarr session ship with the repo."""
        self.assertTrue(COMMITTED_FILES, "no test NWB files found under tests/nwb/")

    def test_backend_dispatch(self):
        """``load_nwb_from_filename`` dispatches to the requested backend."""
        f = COMMITTED_FILES[0]
        direct = nwb_utils.load_nwb_from_filename(f, backend="direct")
        self.assertIsInstance(direct, DirectNWBAdapter)

        auto = nwb_utils.load_nwb_from_filename(f, backend="auto")
        self.assertIsInstance(auto, DirectNWBAdapter)  # "auto" resolves to "direct"

        # An already-loaded object is passed through untouched, regardless of backend.
        self.assertIs(nwb_utils.load_nwb_from_filename(direct, backend="pynwb"), direct)

        with self.assertRaises(ValueError):
            nwb_utils.load_nwb_from_filename(f, backend="not_a_backend")

    def test_adapter_matches_pynwb_metadata(self):
        """The adapter reproduces the scalar metadata pynwb exposes (incl. quirks)."""
        for f in ALL_FILES:
            with self.subTest(file=f):
                pn = nwb_utils.load_nwb_from_filename(f, backend="pynwb")
                dn = nwb_utils.load_nwb_from_filename(f, backend="direct")
                self.assertEqual(pn.session_id, dn.session_id)
                self.assertEqual(pn.protocol, dn.protocol)
                self.assertEqual(pn.subject.subject_id, dn.subject.subject_id)
                # pynwb returns experimenter as a tuple; the adapter must match so
                # that downstream ``nwb.experimenter[0]`` behaves identically.
                self.assertEqual(pn.experimenter, dn.experimenter)
                pd.testing.assert_frame_equal(
                    pn.trials.to_dataframe(), dn.trials.to_dataframe(), check_dtype=False
                )

    def test_behavior_readers_match_pynwb(self):
        """session / trials / events match pynwb on every file (CO assets first)."""
        for f in ALL_FILES:
            for reader_name in BEHAVIOR_READERS:
                with self.subTest(reader=reader_name, file=f):
                    out_pynwb = _run(reader_name, f, "pynwb", **BEHAVIOR_READERS[reader_name])
                    out_direct = _run(reader_name, f, "direct", **BEHAVIOR_READERS[reader_name])
                    _assert_frames_equal(
                        out_pynwb, out_direct, f"{reader_name} differs for {f}"
                    )

    def test_fip_reader_matches_pynwb(self):
        """create_df_fip matches pynwb on the committed files.

        Only the committed sessions are used here: ``create_df_fip`` on a full CO-asset
        session reads dozens of large FIP streams, which is too heavy for a unit test.
        """
        for f in COMMITTED_FILES:
            with self.subTest(file=f):
                out_pynwb = _run("create_df_fip", f, "pynwb", verbose=False)
                out_direct = _run("create_df_fip", f, "direct", verbose=False)
                _assert_frames_equal(out_pynwb, out_direct, f"create_df_fip differs for {f}")

    def test_list_input_uses_backend(self):
        """``create_df_session`` accepts a list of files and honors the backend."""
        df_pynwb = nwb_utils.create_df_session(COMMITTED_FILES, backend="pynwb")
        df_direct = nwb_utils.create_df_session(COMMITTED_FILES, backend="direct")
        self.assertEqual(len(df_direct), len(COMMITTED_FILES))
        pd.testing.assert_frame_equal(
            df_pynwb.reset_index(drop=True),
            df_direct.reset_index(drop=True),
            check_dtype=False,
        )

    def test_speed_comparison(self):
        """Benchmark behaviour-only reading; the direct open must beat pynwb.

        Opening (skipping pynwb's object-graph construction) is the robust, reliably
        faster step, so that is asserted. Full-reader timings are printed for reference.
        """
        files = ALL_FILES
        reps = 1 if CO_ASSET_FILES else 3  # CO assets (esp. S3) are slow; one pass
        totals = {
            "load_nwb (open only)": {"pynwb": 0.0, "direct": 0.0},
            "create_df_trials": {"pynwb": 0.0, "direct": 0.0},
            "create_df_events": {"pynwb": 0.0, "direct": 0.0},
        }
        readers = [
            ("load_nwb (open only)", "load_nwb_from_filename", {}),
            ("create_df_trials", "create_df_trials", dict(verbose=False)),
            ("create_df_events", "create_df_events", dict(verbose=False)),
        ]
        rows = []
        for f in files:
            for label, reader_name, kwargs in readers:
                timings = {}
                for backend in ("pynwb", "direct"):
                    best = float("inf")
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        _run(reader_name, f, backend, **kwargs)
                        best = min(best, time.perf_counter() - t0)
                    timings[backend] = best
                    totals[label][backend] += best
                rows.append(
                    f"  {label:21s} {os.path.basename(f)[:40]:40s} "
                    f"pynwb={timings['pynwb']:.3f}s  direct={timings['direct']:.3f}s  "
                    f"speedup={timings['pynwb'] / timings['direct']:.2f}x"
                )

        print("\n[direct backend speed comparison]")
        print("\n".join(rows))
        for label, t in totals.items():
            print(f"  TOTAL {label:21s} pynwb={t['pynwb']:.3f}s  "
                  f"direct={t['direct']:.3f}s  speedup={t['pynwb'] / t['direct']:.2f}x")

        open_t = totals["load_nwb (open only)"]
        self.assertLess(
            open_t["direct"],
            open_t["pynwb"],
            "direct open should be faster than pynwb (it skips graph construction)",
        )


if __name__ == "__main__":
    unittest.main()
