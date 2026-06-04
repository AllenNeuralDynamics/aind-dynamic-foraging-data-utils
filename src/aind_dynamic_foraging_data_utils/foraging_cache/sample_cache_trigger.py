"""
Build (or incrementally extend) the foraging parquet cache.

Runs the full pipeline over ALL sessions in the Han session table, exercising
all three NWB routes (see references/data-sources.md):

  - CO asset  -> AIND reader (nwb_utils) on the docDB S3 URI
  - bonsai S3 -> legacy reader (Han bonsai NWB)
  - bpod S3   -> legacy reader (Han bpod NWB)

Incremental by default: only sessions not already recorded in
``build_metadata.json`` are processed, so re-running cheaply adds new sessions.
Pass ``--full-rebuild`` to reprocess everything.

Output target:
  - Default is a local scratch directory (safe for dev iteration).
  - Point ``--out-dir`` at the canonical S3 prefix to write the production
    database:  ``--out-dir s3://aind-behavior-data/foraging_cache``

Run:
    # incremental local build (default scratch dir)
    python -m aind_dynamic_foraging_data_utils.foraging_cache.sample_cache_trigger

    # production build/update on S3
    python -m aind_dynamic_foraging_data_utils.foraging_cache.sample_cache_trigger \\
        --out-dir s3://aind-behavior-data/foraging_cache

    # quick smoke test on a random 300-session subset (spans all three routes)
    python -m aind_dynamic_foraging_data_utils.foraging_cache.sample_cache_trigger --limit 300

Or drive it programmatically (the module is import-safe — nothing runs on import):
    from aind_dynamic_foraging_data_utils.foraging_cache import sample_cache_trigger as t
    t.main(t.Config(out_dir="/root/capsule/scratch/tmp/foraging_cache", limit=300))
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

from aind_dynamic_foraging_data_utils.foraging_cache import parquet_builder

logger = logging.getLogger(__name__)

# Default output: local scratch (never /tmp). Use --out-dir for S3 prod.
DEFAULT_OUT_DIR = "/root/capsule/scratch/tmp/foraging_cache"
PROD_S3_OUT_DIR = "s3://aind-behavior-data/foraging_cache"


@dataclass
class Config:
    """Inputs and derived output paths for one build."""

    out_dir: str = DEFAULT_OUT_DIR
    limit: Optional[int] = None  # cap sessions for a quick test (random subset)
    full_rebuild: bool = False  # ignore build metadata; reprocess everything
    random_seed: int = 42

    @property
    def is_s3(self) -> bool:
        return self.out_dir.startswith("s3://")

    @property
    def session_out(self) -> str:
        return f"{self.out_dir}/session_table.parquet"

    @property
    def trial_out(self) -> str:
        return f"{self.out_dir}/trial_table"

    @property
    def event_out(self) -> str:
        return f"{self.out_dir}/event_table"

    @property
    def meta_out(self) -> str:
        return f"{self.out_dir}/build_metadata.json"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def build_session_table(cfg: Config):
    """Build the full session table (all sessions, enriched with CO assets)."""
    _banner("Building session table (Han + docDB CO assets)")
    return parquet_builder.build_session_table(
        output_path=cfg.session_out,
        include_co_assets=True,
        incremental=not cfg.full_rebuild,
        verbose=True,
    )


def build_trial_event_tables(cfg: Config, session_df, nwb_index: dict) -> dict:
    """Build the Hive-partitioned trial + event tables for every session."""
    _banner("Building trial and event tables")
    df = session_df
    if cfg.limit is not None and len(df) > cfg.limit:
        # Random subset so a quick test still spans CO / bonsai / bpod routes.
        df = df.sample(n=cfg.limit, random_state=cfg.random_seed)
        print(f"  --limit: randomly sampled {len(df)} of {len(session_df)} sessions")
    return parquet_builder.build_trial_and_event_tables(
        session_df=df,
        trial_output_prefix=cfg.trial_out,
        event_output_prefix=cfg.event_out,
        nwb_file_index=nwb_index,
        build_metadata_path=cfg.meta_out,
        incremental=not cfg.full_rebuild,
        verbose=True,
    )


def run_example_queries(cfg: Config) -> None:
    """Demonstrate the standard DuckDB query patterns against the built tables."""
    import duckdb

    _banner("EXAMPLE QUERIES")

    read_trials = (
        f"read_parquet('{cfg.trial_out}/**/*.parquet', "
        "hive_partitioning=true, union_by_name=true)"
    )
    read_events = (
        f"read_parquet('{cfg.event_out}/**/*.parquet', "
        "hive_partitioning=true, union_by_name=true)"
    )
    sel_cte = f"""
        WITH sel AS (
            SELECT _session_id, subject_id, session_date, task, foraging_eff
            FROM read_parquet('{cfg.session_out}')
            WHERE task LIKE '%Uncoupled%' AND foraging_eff > 0.8
        )
    """

    print("\n--- Session query: Uncoupled tasks with foraging_eff > 0.8 ---")
    selected = duckdb.sql(f"""
        SELECT _session_id, subject_id, session_date, finished_trials, foraging_eff, task
        FROM read_parquet('{cfg.session_out}')
        WHERE task LIKE '%Uncoupled%' AND foraging_eff > 0.8
        ORDER BY session_date, subject_id
    """).df()
    print(selected.to_string(index=False))

    print(f"\n--- Trial history for {len(selected)} selected sessions ---")
    df_trials_all = duckdb.sql(f"""
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
    print(f"  Total trials across {len(selected)} sessions : {len(df_trials_all)}")
    print(df_trials_all.head(10).to_string(index=False))

    print(f"\n--- Event history for {len(selected)} selected sessions ---")
    df_events_all = duckdb.sql(f"""
        {sel_cte}
        SELECT s.subject_id, s.session_date, e.session_id, e.timestamps, e.event, e.data
        FROM {read_events} e
        JOIN sel s ON e.session_id = s._session_id
        WHERE CAST(e.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
        ORDER BY s.subject_id, s.session_date, e.timestamps
    """).df()
    print(f"  Total events across {len(selected)} sessions : {len(df_events_all)}")
    print(f"  Event types : {sorted(df_events_all['event'].unique().tolist())}")
    print(df_events_all.head(10).to_string(index=False))


def print_summary(cfg: Config, summary: dict) -> None:
    """Print the build-result breakdown."""
    _banner("BUILD SUMMARY")
    print(f"  Output dir                   : {cfg.out_dir}")
    print(f"  Processed (ok)               : {summary['n_processed']}")
    print(f"  Skipped (no NWB found)       : {summary['n_skipped']}")
    print(f"  Failed                       : {summary['n_failed']}")
    print("\n  Data source breakdown:")
    print(f"    CO asset                   : {summary['n_co_asset']}")
    print(f"    Bonsai S3                  : {summary['n_bonsai_s3']}")
    print(f"    bpod S3                    : {summary['n_bpod_s3']}")
    print("\n  NWB reader breakdown:")
    print(f"    AIND reader (CO asset)     : {summary['n_aind_reader']}")
    print(f"    AIND->legacy fallback      : {summary['n_aind_fallback_legacy']}")
    print(f"    Legacy bonsai              : {summary['n_legacy_bonsai']}")
    print(f"    Legacy bpod                : {summary['n_legacy_bpod']}")
    if summary["failed_sessions"]:
        print(f"\n  Failed sessions ({summary['n_failed']}), first 20:")
        for fs in summary["failed_sessions"][:20]:
            print(f"    [{fs.get('data_source', '?')}] {fs['session_id']}  --  {fs['error']}")
        if summary["n_failed"] > 20:
            print(f"    ... and {summary['n_failed'] - 20} more")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main(cfg: Config) -> dict:
    """Run the full build pipeline end to end. Returns the build summary."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not cfg.is_s3:
        os.makedirs(cfg.out_dir, exist_ok=True)

    _banner("Indexing local Han NWB files")
    nwb_index = parquet_builder.build_nwb_file_index()
    print(f"  Total NWB files indexed: {len(nwb_index)}")

    session_df = build_session_table(cfg)
    summary = build_trial_event_tables(cfg, session_df, nwb_index)
    print_summary(cfg, summary)
    run_example_queries(cfg)
    return summary


def parse_args(argv=None) -> Config:
    """Parse CLI arguments into a Config."""
    p = argparse.ArgumentParser(description="Build/extend the foraging parquet cache.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"output dir or S3 prefix (default: %(default)s; "
                        f"prod: {PROD_S3_OUT_DIR})")
    p.add_argument("--limit", type=int, default=None,
                   help="cap to a random N-session subset for a quick test (default: all)")
    p.add_argument("--full-rebuild", action="store_true",
                   help="ignore build metadata and reprocess every session")
    args = p.parse_args(argv)
    return Config(out_dir=args.out_dir, limit=args.limit, full_rebuild=args.full_rebuild)


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
