"""
Query examples / read-back ("return loop") for the foraging parquet cache.

This is the counterpart to ``build_cache`` — it does NOT build anything, it only
reads a built cache back and demonstrates the standard DuckDB query patterns
(session filter -> join trials -> join events). Point ``--out-dir`` at any cache
built by ``build_cache`` (local dir or ``s3://`` prefix); DuckDB reads S3 natively.

Run:
    # against the production S3 cache
    python -m aind_dynamic_foraging_data_utils.foraging_cache.query_examples \\
        --out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache

    # against a local build
    python -m aind_dynamic_foraging_data_utils.foraging_cache.query_examples \\
        --out-dir /root/capsule/scratch/tmp/foraging_cache

Programmatic use:
    from aind_dynamic_foraging_data_utils.foraging_cache import query_examples as q
    dfs = q.run("s3://aind-scratch-data/aind-dynamic-foraging-cache")
    dfs["trials"], dfs["events"], dfs["sessions"]
"""

import argparse


def run(out_dir, task_like="%Uncoupled%", min_foraging_eff=0.8, verbose=True):
    """
    Run the standard session -> trials/events read-back against a built cache.

    Reads from ``{out_dir}/session_table.parquet`` and the Hive-partitioned
    ``trial_table/`` and ``event_table/``. Returns a dict of DataFrames:
    ``{"sessions": ..., "trials": ..., "events": ...}``.
    """
    import duckdb

    session_out = f"{out_dir}/session_table.parquet"
    read_trials = (
        f"read_parquet('{out_dir}/trial_table/**/*.parquet', "
        "hive_partitioning=true, union_by_name=true)"
    )
    read_events = (
        f"read_parquet('{out_dir}/event_table/**/*.parquet', "
        "hive_partitioning=true, union_by_name=true)"
    )
    sel_cte = f"""
        WITH sel AS (
            SELECT _session_id, subject_id, session_date, task, foraging_eff
            FROM read_parquet('{session_out}')
            WHERE task LIKE '{task_like}' AND foraging_eff > {min_foraging_eff}
        )
    """

    # 1. Session-level filter
    sessions = duckdb.sql(f"""
        SELECT _session_id, subject_id, session_date, finished_trials, foraging_eff, task
        FROM read_parquet('{session_out}')
        WHERE task LIKE '{task_like}' AND foraging_eff > {min_foraging_eff}
        ORDER BY session_date, subject_id
    """).df()

    # 2. Trial history for the selected sessions (session keys merged in)
    trials = duckdb.sql(f"""
        {sel_cte}
        SELECT s.subject_id, s.session_date, s.task, s.foraging_eff,
               t.session_id, t.animal_response, t.earned_reward,
               t.reward_probabilityL, t.reward_probabilityR,
               t.rewarded_historyL, t.rewarded_historyR
        FROM {read_trials} t
        JOIN sel s ON t.session_id = s._session_id
        WHERE CAST(t.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
        ORDER BY s.subject_id, s.session_date
    """).df()

    # 3. Event history for the selected sessions
    events = duckdb.sql(f"""
        {sel_cte}
        SELECT s.subject_id, s.session_date, e.session_id, e.timestamps, e.event, e.data
        FROM {read_events} e
        JOIN sel s ON e.session_id = s._session_id
        WHERE CAST(e.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
        ORDER BY s.subject_id, s.session_date, e.timestamps
    """).df()

    if verbose:
        print(f"--- Sessions: task LIKE '{task_like}', foraging_eff > {min_foraging_eff} ---")
        print(sessions.to_string(index=False))
        print(f"\n--- Trials across {len(sessions)} sessions: {len(trials)} rows ---")
        print(trials.head(10).to_string(index=False))
        print(f"\n--- Events across {len(sessions)} sessions: {len(events)} rows ---")
        if len(events):
            print(f"  event types: {sorted(events['event'].unique().tolist())}")
        print(events.head(10).to_string(index=False))

    return {"sessions": sessions, "trials": trials, "events": events}


def main(argv=None):
    p = argparse.ArgumentParser(description="Read-back query examples for the foraging cache.")
    p.add_argument("--out-dir", required=True,
                   help="cache location (local dir or s3:// prefix) built by build_cache")
    p.add_argument("--task-like", default="%Uncoupled%",
                   help="SQL LIKE pattern for the task column (default: %(default)s)")
    p.add_argument("--min-foraging-eff", type=float, default=0.8,
                   help="minimum foraging efficiency (default: %(default)s)")
    args = p.parse_args(argv)
    run(args.out_dir, task_like=args.task_like, min_foraging_eff=args.min_foraging_eff)


if __name__ == "__main__":
    main()
