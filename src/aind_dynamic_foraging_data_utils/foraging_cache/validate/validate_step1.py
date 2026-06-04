"""
Validation step 1: data-equivalence + performance of the parquet cache vs the
TRUE legacy route documented in the repo README.

Legacy route per CO/AIND session = the full chain a user runs today:
  1. code_ocean_utils.get_subject_assets(subject)   -- docDB query for the asset
  2. code_ocean_utils.add_s3_location(row)           -- recursive S3 glob for the NWB
  3. nwb_utils.create_df_trials / create_df_events   -- open + parse the NWB
Each data type (5col / full / full+event) COLD-loads — the whole chain is rerun
(legacy has no column projection: even "5 cols" reads the whole NWB). For Han
bonsai/bpod sources there is no docDB asset, so the legacy route is open-local +
legacy reader. Legacy is measured at small N (1, 10) per source and EXTRAPOLATED
(strictly O(N), far too slow to run at 1k+).

CACHE timing is MEASURED at every scale (1/10/100/1k/10k/full); the S3 connection
is warmed up once so the curve isn't polluted by a cold start.

Artifacts -> scratch/tmp/validation/.
"""
import os, time, random, warnings, re
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, duckdb

from aind_dynamic_foraging_data_utils import nwb_utils
import aind_dynamic_foraging_data_utils.code_ocean_utils as cou
from aind_dynamic_foraging_data_utils.foraging_cache.util import nwb_reader_legacy, parquet_builder

OUT = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
TRIALS = f"read_parquet('{OUT}/trial_table/**/*.parquet', hive_partitioning=true, union_by_name=true)"
EVENTS = f"read_parquet('{OUT}/event_table/**/*.parquet', hive_partitioning=true, union_by_name=true)"
ART = "/root/capsule/scratch/tmp/validation"; os.makedirs(f"{ART}/dataframes", exist_ok=True)
SOURCES = ["co_asset", "bonsai_s3", "bpod_s3"]
KEY = ["animal_response", "earned_reward", "reward_probabilityL", "reward_probabilityR"]
FIVE = "animal_response, earned_reward, reward_probabilityL, reward_probabilityR"
OPS = ["trial_5col", "trial_full", "trial+event_full"]


def session_name(sid):
    s, d, suf = sid.split("_"); t = f"{int(suf):06d}"
    return f"behavior_{s}_{d}_{t[:2]}-{t[2:4]}-{t[4:]}"


def main():
    sess = pd.read_parquet(f"{OUT}/session_table.parquet")
    log = pd.read_csv(f"{OUT}/processing_log.csv")
    sess = sess[sess["_session_id"].isin(set(log[log.status == "ok"]["session_id"]))].copy()
    all_ids = sorted(sess["_session_id"]); n_full = len(all_ids)
    rng = random.Random(42)
    per_src = {(s, N): rng.sample(sorted(sess[sess.nwb_data_source == s]["_session_id"]), N)
               for s in SOURCES for N in [1, 10]}
    scale = {N: rng.sample(all_ids, N) for N in [1, 10, 100, 1000, 10000]}
    idx = parquet_builder.build_nwb_file_index()
    row_of = {sid: sess[sess._session_id == sid].iloc[0]
              for s in SOURCES for N in [1, 10] for sid in per_src[(s, N)]}

    def fetch(ids, op):
        duckdb.register("want", pd.DataFrame({"sid": list(ids)}))
        if op == "trial_5col":
            return duckdb.sql(f"SELECT t.session_id, {FIVE} FROM {TRIALS} t JOIN want w ON t.session_id=w.sid").df()
        if op == "trial_full":
            return duckdb.sql(f"SELECT t.* FROM {TRIALS} t JOIN want w ON t.session_id=w.sid").df()
        td = duckdb.sql(f"SELECT t.* FROM {TRIALS} t JOIN want w ON t.session_id=w.sid").df()
        ed = duckdb.sql(f"SELECT e.* FROM {EVENTS} e JOIN want w ON e.session_id=w.sid").df()
        return td, ed

    def fetch_full(op):
        if op == "trial_5col":
            return duckdb.sql(f"SELECT session_id, {FIVE} FROM {TRIALS}").df()
        return duckdb.sql(f"SELECT * FROM {TRIALS}").df()

    # ---- CACHE timing (measured) ----
    print("== cache timing (measured) ==", flush=True)
    duckdb.sql(f"SELECT 1 FROM {TRIALS} LIMIT 1").df()  # warm up S3 connection
    rows = []
    for N in [1, 10, 100, 1000, 10000]:
        for op in OPS:
            t = time.time(); r = fetch(scale[N], op); sec = time.time() - t
            rows.append(dict(N=N, op=op, sec=sec, rows=(len(r[0]) + len(r[1])) if isinstance(r, tuple) else len(r)))
        print(f"  N={N} done", flush=True)
    for op in ["trial_5col", "trial_full"]:
        t = time.time(); r = fetch_full(op); rows.append(dict(N=n_full, op=op, sec=time.time() - t, rows=len(r)))
        print(f"  full {op} done", flush=True)
    cache = pd.DataFrame(rows); cache.to_csv(f"{ART}/cache_timing.csv", index=False)

    # ---- LEGACY: TRUE chain (docDB query -> S3 glob -> open), cold per data type ----
    print("== legacy TRUE chain (get_subject_assets -> add_s3_location -> create_df_*) ==", flush=True)

    def legacy_one(sid, src, want_events):
        """Full legacy chain for one session; returns dict of per-step seconds."""
        row = row_of[sid]
        if src == "co_asset":
            subj = sid.split("_")[0]; sn = session_name(sid)
            t = time.time(); res = cou.get_subject_assets(subj); t_doc = time.time() - t
            match = res[res["session_name"] == sn] if (res is not None and "session_name" in res) else None
            if match is None or len(match) == 0:
                match = res.iloc[[0]] if res is not None and len(res) else None
            t = time.time(); match = cou.add_s3_location(match.copy()); t_glob = time.time() - t
            uri = match["s3_nwb_location"].iloc[0]
            t = time.time(); nwb_utils.create_df_trials(uri, verbose=False); t_tr = time.time() - t
            t_ev = 0.0
            if want_events:
                t = time.time(); nwb_utils.create_df_events(uri, verbose=False); t_ev = time.time() - t
            return dict(docDB=t_doc, glob=t_glob, open_trials=t_tr, open_events=t_ev)
        else:  # bonsai/bpod: no docDB asset -> open local NWB + legacy reader
            s, d, suf = sid.split("_"); p = idx[(s, d, int(suf))]
            t = time.time(); nwb_reader_legacy.read_trials(p); t_tr = time.time() - t
            t_ev = 0.0
            if want_events:
                t = time.time(); nwb_reader_legacy.read_events(p); t_ev = time.time() - t
            return dict(docDB=0.0, glob=0.0, open_trials=t_tr, open_events=t_ev)

    lrows = []
    for src in SOURCES:
        for N in [1, 10]:
            ids = per_src[(src, N)]
            # cold pass for trials (covers 5col == full: no projection in legacy)
            comp = [legacy_one(sid, src, want_events=False) for sid in ids]
            sec_tr = sum(c["docDB"] + c["glob"] + c["open_trials"] for c in comp)
            # cold pass for trials+events (full chain rerun + events)
            comp2 = [legacy_one(sid, src, want_events=True) for sid in ids]
            sec_te = sum(c["docDB"] + c["glob"] + c["open_trials"] + c["open_events"] for c in comp2)
            for op, s in [("trial_5col", sec_tr), ("trial_full", sec_tr), ("trial+event_full", sec_te)]:
                lrows.append(dict(source=src, N=N, op=op, sec=s, per_session=s / N))
            agg = pd.DataFrame(comp + comp2).mean()
            print(f"  {src} N={N}: trial={sec_tr/N:.1f}s/sess trial+evt={sec_te/N:.1f}s/sess "
                  f"[docDB {agg.docDB:.1f} | glob {agg.glob:.1f} | openT {agg.open_trials:.1f} | openE {agg.open_events:.1f}]",
                  flush=True)
    legacy = pd.DataFrame(lrows); legacy.to_csv(f"{ART}/legacy_timing.csv", index=False)

    # ---- correctness (reuse direct read = legacy chain result vs cache) ----
    print("== correctness ==", flush=True)
    def eq(a, b):
        a, b = a.reset_index(drop=True), b.reset_index(drop=True)
        if len(a) != len(b): return False
        try: return np.allclose(np.sort(pd.to_numeric(a, errors="coerce")),
                                 np.sort(pd.to_numeric(b, errors="coerce")), equal_nan=True)
        except Exception: return sorted(a.astype(str)) == sorted(b.astype(str))

    def direct_trials(sid, src):
        if src == "co_asset":
            subj = sid.split("_")[0]; sn = session_name(sid)
            res = cou.get_subject_assets(subj); m = res[res.session_name == sn]
            m = cou.add_s3_location((m if len(m) else res.iloc[[0]]).copy())
            try: return nwb_utils.create_df_trials(m["s3_nwb_location"].iloc[0], verbose=False)
            except Exception: return nwb_reader_legacy.read_trials(m["s3_nwb_location"].iloc[0])
        s, d, suf = sid.split("_"); return nwb_reader_legacy.read_trials(idx[(s, d, int(suf))])

    corr = []
    for src in SOURCES:
        ids = per_src[(src, 1)] + per_src[(src, 10)]
        cach = fetch(ids, "trial_full")
        for sid in ids:
            c = cach[cach.session_id == sid]
            try:
                direct = direct_trials(sid, src)
                rec = dict(source=src, session=sid, n_cache=len(c), n_direct=len(direct),
                           count_match=(len(c) == len(direct)))
                for k in KEY:
                    rec[k] = eq(c[k], direct[k]) if (k in c and k in direct) else None
                corr.append(rec)
            except Exception as e:
                corr.append(dict(source=src, session=sid, count_match=False, error=str(e)[:80]))
    corr = pd.DataFrame(corr); corr.to_csv(f"{ART}/correctness.csv", index=False)
    kc = [k for k in KEY if k in corr.columns]
    corr["all_match"] = corr["count_match"] & corr[kc].fillna(True).all(axis=1)
    print(f"  EXACT-MATCH: {corr.all_match.sum()}/{len(corr)}", flush=True)

    # ---- figure (delegated to plot_validation: explicit axis limits, no
    #      bbox_inches='tight', which otherwise blows up the canvas height under
    #      the wide log-scale autoscale of the legacy extrapolation) ----
    print("== figure ==", flush=True)
    from aind_dynamic_foraging_data_utils.foraging_cache.validate import plot_validation
    plot_validation.main()
    print("\nLEGACY per-session (cold full chain):\n" +
          legacy.groupby(["source", "op"]).per_session.mean().round(1).to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
