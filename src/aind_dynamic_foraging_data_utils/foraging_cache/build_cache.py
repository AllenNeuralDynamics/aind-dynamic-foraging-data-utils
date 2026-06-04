"""
Main entry point to build (or incrementally extend) the foraging parquet cache.

Runs the full pipeline over ALL sessions in the Han session table, exercising
all three NWB routes (see references/data-sources.md):

  - CO asset  -> AIND reader (nwb_utils) on the docDB S3 URI
  - bonsai S3 -> legacy reader (Han bonsai NWB)
  - bpod S3   -> legacy reader (Han bpod NWB)

Incremental by default: only sessions not already recorded in
``build_metadata.json`` are processed, so re-running cheaply adds new sessions.
Pass ``--full-rebuild`` to reprocess everything.

To query the built cache (the read-back / "return loop"), use the companion
``query_examples`` module or ``query_examples.ipynb`` — querying is intentionally
kept out of this build script.

Output target:
  - Default is a local scratch directory (safe for dev iteration).
  - Point ``--out-dir`` at the canonical S3 prefix to write the production
    database:  ``--out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache``

Run:
    # incremental local build (default scratch dir)
    python -m aind_dynamic_foraging_data_utils.foraging_cache.build_cache

    # production build/update on S3 (--n-workers 64 ~= 4x faster; see --help)
    python -m aind_dynamic_foraging_data_utils.foraging_cache.build_cache \\
        --out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache --n-workers 64

    # quick smoke test on a random 300-session subset (spans all three routes)
    python -m aind_dynamic_foraging_data_utils.foraging_cache.build_cache --limit 300

Or drive it programmatically (the module is import-safe — nothing runs on import):
    from aind_dynamic_foraging_data_utils.foraging_cache import build_cache as b
    b.main(b.Config(out_dir="/root/capsule/scratch/tmp/foraging_cache", limit=300))
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
PROD_S3_OUT_DIR = "s3://aind-scratch-data/aind-dynamic-foraging-cache"


@dataclass
class Config:
    """Inputs and derived output paths for one build."""

    out_dir: str = DEFAULT_OUT_DIR
    limit: Optional[int] = None  # cap sessions for a quick test (random subset)
    full_rebuild: bool = False  # ignore build metadata; reprocess everything
    random_seed: int = 42
    n_workers: Optional[int] = None  # worker processes; None -> CO_CPUS-1
    coalesce: bool = True  # merge each subject's sessions into one parquet file
    co_cache: Optional[str] = None  # dev: cache the docDB discovery parquet here

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

    @property
    def log_csv(self) -> str:
        return f"{self.out_dir}/processing_log.csv"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def build_sessions(cfg: Config):
    """
    Build the complete session table: Han ∪ docDB/CO universe, CO assets attached.

    See parquet_builder.build_session_table / _merge_han_and_co for the match
    rule. The ~137 s docDB discovery can be cached locally for dev iteration via
    --co-cache (loaded if present, else fetched once and saved).
    """
    _banner("Building session table (Han ∪ docDB CO universe)")
    return parquet_builder.build_session_table(
        output_path=cfg.session_out,
        include_co_assets=True,
        co_discovery=_load_or_fetch_co_discovery(cfg),
        verbose=True,
    )


def _load_or_fetch_co_discovery(cfg: Config):
    """
    Dev helper: if --co-cache is set, load the cached docDB discovery parquet (or
    fetch once and save it). Returns None when no cache is configured, so
    build_session_table fetches fresh.
    """
    if not cfg.co_cache:
        return None
    import pandas as pd

    if os.path.exists(cfg.co_cache):
        print(f"  using cached docDB discovery: {cfg.co_cache}")
        return pd.read_parquet(cfg.co_cache)

    from aind_dynamic_foraging_data_utils.code_ocean_utils import get_dynamic_foraging_assets

    print(f"  fetching docDB discovery, caching -> {cfg.co_cache}")
    co = get_dynamic_foraging_assets()
    cols = ["name", "session_name", "location", "code_ocean_asset_id", "subject_id"]
    co[[c for c in cols if c in co.columns]].to_parquet(cfg.co_cache)
    return co


def select_sessions(cfg: Config, session_df):
    """
    Optionally subsample the complete session table for a quick test (--limit N,
    seeded). The full table is still written to session_out by build_sessions;
    only the trial/event build is limited.
    """
    if cfg.limit is None or len(session_df) <= cfg.limit:
        return session_df
    sampled = session_df.sample(n=cfg.limit, random_state=cfg.random_seed)
    print(f"  --limit: randomly sampled {len(sampled)} of {len(session_df)} sessions")
    return sampled.reset_index(drop=True)


def build_trial_event_tables(cfg: Config, session_df, nwb_index: dict) -> dict:
    """Build the Hive-partitioned trial + event tables for the selected sessions."""
    _banner("Building trial and event tables")
    return parquet_builder.build_trial_and_event_tables(
        session_df=session_df,
        trial_output_prefix=cfg.trial_out,
        event_output_prefix=cfg.event_out,
        nwb_file_index=nwb_index,
        build_metadata_path=cfg.meta_out,
        incremental=not cfg.full_rebuild,
        n_workers=cfg.n_workers,
        coalesce=cfg.coalesce,
        log_csv_path=cfg.log_csv,
        verbose=True,
    )


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

    # Build the complete session table (Han ∪ CO universe, CO assets attached),
    # then optionally subsample for a quick --limit test, then build the tables.
    session_df = build_sessions(cfg)
    session_df = select_sessions(cfg, session_df)

    summary = build_trial_event_tables(cfg, session_df, nwb_index)
    print_summary(cfg, summary)
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
    p.add_argument("--n-workers", type=int, default=None,
                   help="worker processes (default: CO_CPUS-1). CO-asset reads are "
                        "I/O-bound, so oversubscribing past CPU count overlaps S3 "
                        "latency. Recommended ~64 on a 16-core box: ~4x faster than "
                        "the default, and beyond ~64 there's no gain (the "
                        "create_df_trials parse saturates the cores). RAM is not "
                        "the limit (~21 GB at 128 workers).")
    p.add_argument("--no-coalesce", action="store_true",
                   help="keep one parquet file per session instead of merging each "
                        "subject's sessions into a single sorted file (the default)")
    p.add_argument("--co-cache", default=None,
                   help="dev: parquet path to cache the ~137s docDB discovery "
                        "(loaded if present, else fetched once and saved)")
    args = p.parse_args(argv)
    return Config(
        out_dir=args.out_dir,
        limit=args.limit,
        full_rebuild=args.full_rebuild,
        n_workers=args.n_workers,
        coalesce=not args.no_coalesce,
        co_cache=args.co_cache,
    )


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
