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


def build_session_pool(cfg: Config):
    """
    Build the Han session table WITHOUT CO assets — fast (no docDB / no S3).

    This is the pool we sample from. CO-asset enrichment (the slow docDB + S3
    step) is deliberately deferred until after sampling, so a --limit run only
    pays that cost for the sampled sessions.
    """
    _banner("Building session pool (Han metadata only — no CO assets yet)")
    return parquet_builder.build_session_table(
        output_path=cfg.session_out,
        include_co_assets=False,
        incremental=not cfg.full_rebuild,
        verbose=True,
    )


def select_sessions(cfg: Config, session_df):
    """
    Pick the sessions to build EARLY, from Han metadata, before any AIND/CO work.

    Full pool unless --limit N is given, in which case a random N-session subset
    (seeded) — which still spans CO / bonsai / bpod routes.
    """
    if cfg.limit is None or len(session_df) <= cfg.limit:
        return session_df
    sampled = session_df.sample(n=cfg.limit, random_state=cfg.random_seed)
    print(f"  --limit: randomly sampled {len(sampled)} of {len(session_df)} sessions (early)")
    return sampled.reset_index(drop=True)


def enrich_selected(cfg: Config, session_df):
    """
    Enrich ONLY the selected sessions with CO assets (docDB + S3 location).

    Reuses the builder internals so the slow S3 globbing touches only the
    sampled sessions, then rewrites the session table with the CO columns.
    """
    _banner(f"Enriching {len(session_df)} selected sessions with CO assets (docDB + S3)")
    subjects = sorted(session_df["subject_id"].astype(str).unique().tolist())
    print(f"  Unique subjects in selection: {len(subjects)}")

    session_df = parquet_builder._enrich_with_co_assets(session_df, subjects, verbose=True)
    # Refresh the preferred NWB source now that CO assets are populated.
    session_df["nwb_data_source"] = session_df.apply(
        parquet_builder._assign_nwb_data_source, axis=1
    )
    # Rewrite the session table so it reflects only the (enriched) selection.
    parquet_builder._write_dataframe_as_parquet(session_df, cfg.session_out)

    has_co = session_df["co_s3_nwb_uri"].notna() & (session_df["co_s3_nwb_uri"] != "")
    print(f"  With CO asset : {has_co.sum()}   Local only : {(~has_co).sum()}")
    return session_df


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

    # Sample EARLY from Han metadata, BEFORE any docDB/S3 (CO-asset) work, so a
    # --limit run only enriches + S3-globs the sampled sessions.
    session_df = build_session_pool(cfg)
    session_df = select_sessions(cfg, session_df)
    session_df = enrich_selected(cfg, session_df)

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
    args = p.parse_args(argv)
    return Config(
        out_dir=args.out_dir,
        limit=args.limit,
        full_rebuild=args.full_rebuild,
        n_workers=args.n_workers,
        coalesce=not args.no_coalesce,
    )


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
