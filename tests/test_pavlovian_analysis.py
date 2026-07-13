"""Tests for aind_pavlovian_data_utils.pavlovian_analysis."""

import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
from dateutil.tz import tzutc
from hdmf.common import DynamicTable, VectorData
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject

from aind_pavlovian_data_utils import pavlovian_analysis as pa

MS = 1000.0


def _build_pavlovian_nwb(cs_types, path, sr=20.0, dur=400.0, seed=0):
    """Write a minimal synthetic Pavlovian NWB (ms clock) to ``path``."""
    nwb = NWBFile(
        session_description="pav",
        identifier="pav-test",
        session_start_time=datetime(2025, 8, 21, 12, 12, 44, tzinfo=tzutc()),
        session_id="behavior_774212_2025-08-21_12-12-44",
    )
    nwb.subject = Subject(subject_id="774212")
    n = int(dur * sr)
    ts_ms = (np.arange(n) / sr) * MS
    fp = nwb.create_processing_module("fiber_photometry", "Fiber photometry data")
    rng = np.random.default_rng(seed)
    for chan in ("G", "Iso", "R"):
        for roi in range(4):
            for variant in ("dff-bright", "dff-bright_mc-iso-IRLS"):
                fp.add(
                    TimeSeries(
                        name=f"{chan}_{roi}_{variant}",
                        data=rng.standard_normal(n) * 0.01,
                        unit="dff",
                        timestamps=ts_ms,
                    )
                )
    nwb.add_trial_column(name="CS_start_time", description="CS onset (ms)")
    ts, names, trials = [], [], []
    onset, trial = 20.0, 0
    for cs in cs_types * 6:
        nwb.add_trial(start_time=onset, stop_time=onset + 13.0, CS_start_time=onset * MS)
        ts.append(onset * MS)
        names.append(cs)
        trials.append(trial)
        if rng.random() < 0.7:
            ts.append((onset + 2.2) * MS)
            names.append("airpuff" if cs == "CS4" else "reward")
            trials.append(trial)
        for k in range(3):
            ts.append((onset + 2.3 + 0.12 * k) * MS)
            names.append("lick")
            trials.append(-1)
        onset += 15.0
        trial += 1
    order = np.argsort(ts)
    table = DynamicTable(
        name="pavlovian_events_table",
        description="events",
        columns=[
            VectorData(name="timestamp", description="ms", data=np.array(ts)[order].tolist()),
            VectorData(name="events", description="e", data=np.array(names)[order].tolist()),
            VectorData(name="trial", description="t", data=np.array(trials, float)[order].tolist()),
        ],
    )
    nwb.create_processing_module("pavlovian_events", "Pavlovian events").add(table)
    with NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    return path


class TestPavlovianAnalysis(unittest.TestCase):
    """Exercise paradigm detection, classification, and the PDF pipeline."""

    def test_canonical_event_name(self):
        """Event strings map to the expected canonical keys."""
        self.assertEqual(pa.canonical_event_name("CS1"), "CS1")
        self.assertEqual(pa.canonical_event_name("cs_2"), "CS2")
        self.assertEqual(pa.canonical_event_name("reward"), "Reward")
        self.assertEqual(pa.canonical_event_name("Airpuff"), "Airpuff")
        self.assertEqual(pa.canonical_event_name("left lick"), "Lick")
        self.assertIsNone(pa.canonical_event_name("iti"))

    def test_detect_and_counts(self):
        """All four stages detect correctly with sane trial counts."""
        cases = {
            "Stage1": (["CS3"], "Stage1"),
            "Stage2": (["CS1", "CS2", "CS3"], "Stage2"),
            "Stage0": (["CS3", "CS4"], "Stage0"),
            "Stage3": (["CS1", "CS2", "CS3", "CS4"], "Stage3"),
        }
        with tempfile.TemporaryDirectory() as d:
            for label, (cs_types, expect) in cases.items():
                path = _build_pavlovian_nwb(cs_types, os.path.join(d, f"{label}.nwb"))
                df_events, df_fip, meta = pa.load_pavlovian_dfs(path)
                paradigm = pa.detect_paradigm(df_events)
                self.assertTrue(paradigm["stage"].startswith(expect))
                cls = pa.classify_trials(df_events, paradigm["cs_list"])
                for cs in paradigm["cs_list"]:
                    pm = cls[cs]["pos_mask"]
                    self.assertEqual(len(pm), 6)
                    self.assertEqual(int(pm.sum()) + int((~pm).sum()), 6)

    def test_analyze_nwb_writes_pdf(self):
        """analyze_nwb writes a multi-page PDF and returns a summary."""
        with tempfile.TemporaryDirectory() as d:
            path = _build_pavlovian_nwb(["CS1", "CS2", "CS3", "CS4"], os.path.join(d, "s3.nwb"))
            pdf = os.path.join(d, "summary.pdf")
            summary = pa.analyze_nwb(path, save_path=pdf, plot_types=["all_sess"])
            self.assertTrue(os.path.exists(pdf))
            self.assertEqual(summary["n_roi"], 4)
            self.assertEqual(len(summary["cs"]), 4)

    def test_channels_filter(self):
        """A channels filter restricts channels and ROIs analyzed."""
        with tempfile.TemporaryDirectory() as d:
            path = _build_pavlovian_nwb(["CS1", "CS2", "CS3"], os.path.join(d, "f.nwb"))
            summary = pa.analyze_nwb(
                path, channels={"G_0": "a", "R_0": "b"}, plot_types=["session"]
            )
            self.assertEqual(set(summary["channels"]), {"Green", "Red"})
            self.assertEqual(summary["n_roi"], 1)


if __name__ == "__main__":
    unittest.main()
