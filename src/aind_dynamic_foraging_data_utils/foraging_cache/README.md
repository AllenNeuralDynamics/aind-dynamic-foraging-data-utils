# `foraging_cache` — dynamic-foraging parquet cache

Builds and queries a Hive-partitioned **parquet database** of AIND dynamic-foraging
behavioral data (one row per session / trial / event), assembled from NWB files
across three sources. It exists so analysts can pull behavior for arbitrary mice /
sessions — or the whole dataset — in seconds with DuckDB/pandas, instead of opening
thousands of NWBs.

---

## Quick start

```bash
# Build / incrementally extend the production cache on S3 (recommended workers: 64)
python -m aind_dynamic_foraging_data_utils.foraging_cache.build_cache \
    --out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache --n-workers 64

# Quick local test on a random 1000-session subset (caches the docDB discovery)
python -m aind_dynamic_foraging_data_utils.foraging_cache.build_cache \
    --limit 1000 --n-workers 64 \
    --out-dir /root/capsule/scratch/tmp/foraging_cache \
    --co-cache /root/capsule/scratch/tmp/co_discovery.parquet

# Read it back ("return loop") — works on a local dir or an s3:// prefix
python -m aind_dynamic_foraging_data_utils.foraging_cache.query_examples \
    --out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache
```

---

## Output layout

```
<out_dir>/                                   # local dir or s3://… prefix
  session_table.parquet                      # one row per session (the complete universe)
  trial_table/subject_id=<id>/<subject_id>.parquet   # Hive-partitioned; coalesced 1 file/subject
  event_table/subject_id=<id>/<subject_id>.parquet   # Hive-partitioned; coalesced 1 file/subject
  build_metadata.json                        # processed session IDs (incremental trial/event build)
  processing_log.csv                         # human-readable per-session triage log
  co_skipped_sessions.csv                    # multi-session-per-day CO rows we skipped (issue #146)
```

- **Prod S3 prefix:** `s3://aind-scratch-data/aind-dynamic-foraging-cache/`
- Session key in the session table: `_session_id`; in trial/event tables: `session_id`
  (both = `"{subject_id}_{session_date}_{nwb_suffix}"`).

---

## Module structure

```
foraging_cache/
├── build_cache.py        # ENTRY POINT: build / incrementally extend the cache
├── query_examples.py     # ENTRY POINT: read-back ("return loop") DuckDB query patterns
├── query_examples.ipynb  # notebook version of the queries
└── util/
    ├── parquet_builder.py    # session table (Han∪CO union) + trial/event build + routing
    ├── nwb_reader_aind.py    # AIND reader wrapper (nwb_utils) → AINDReaderQualityError
    ├── nwb_reader_legacy.py  # Han-pipeline reader (bonsai/bpod) + h5py fallback
    └── postprocess.py        # per-subject coalescing + triage CSV log + CSV I/O
```

The query API is **DuckDB directly on the parquet** (there is no `cache_utils.py`).

---

## How a build works (`build_cache.main`)

1. **Index local NWBs** — scan `/data/foraging_nwb_bonsai/` + `/data/foraging_nwb_bpod/`.
2. **Build the session table** (`build_session_table`): union two universes (see below),
   write `session_table.parquet` + `co_skipped_sessions.csv`.
3. **(optional) `--limit N`** — randomly subsample N sessions for a quick test.
4. **Build trial + event tables** (`build_trial_and_event_tables`): parallel per-session
   read → write Hive-partitioned parquet → coalesce per subject → append triage log.

---

## Key design decisions

### 1. Two session universes, unioned
- **Han pipeline session table** — the source of behavioral metadata (`foraging_eff`,
  `finished_trials`, curriculum, …). Still running, but slated for shutdown, and it
  never captured some sessions.
- **docDB / Code Ocean universe** — every processed dynamic-foraging session
  (`get_dynamic_foraging_assets`). The complete CO set, which fills Han's gaps.

We union them so the cache is complete and keeps working after Han's pipeline retires.

### 2. docDB discovery query (`code_ocean_utils.get_dynamic_foraging_assets`)
- Filters on the **task software name** (`session.*.software.name == "dynamic-foraging-task"`)
  and `data_description.data_level == "derived"` — the canonical "is this a processed
  dynamic-foraging session" signal (more reliable than a `_processed_` name regex).
- Keeps only the **behavior-processed** asset per session (`<session_name>_processed_…`),
  dropping other-pipeline derived assets (PoseTracking, opto-sorting, …).
- Uses **server-side pagination** (`paginate=True, paginate_batch_size=5000`) — the result
  is ~19k records and a single unpaginated query is unreliable.

### 3. Han ⇄ CO match rule (`_merge_han_and_co`, see [issue #146](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils/issues/146))
Han is **≤ 1 session per `(subject, date)`** — its `add_session_number()` keeps only the
NWB with the most `finished_trials` per mouse/day ([process_nwbs.py#L569-L573](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/blob/0b63a460682b8b497be5b9c14329e4520c92ffda/code/process_nwbs.py#L569-L573)).
docDB's `HH-MM-SS` often **drifts** from Han's for the same session, so:

- **CO single-session-per-day** → match Han by **`(subject, date)`** (tolerates the drift),
  or add as a **new CO-only row** if that day isn't in Han.
- **CO multi-session-per-day** → keep only the one with an **exact `(subject, date, nwb_suffix)`**
  match to Han (verified unambiguous: never >1 exact match); **log + skip** the rest to
  `co_skipped_sessions.csv` (we can't tell which is "the" session). This preserves genuinely
  separate same-day sessions only when Han confirms them.

> Matching on `(subject, date)` for the single-session case also fixed ~550 sessions whose CO
> asset the old exact-suffix match was missing.

### 4. NWB reader routing (`_read_session_with_fallback`)
| Source | Reader |
|---|---|
| CO asset (docDB S3 URI) | **AIND reader** (`nwb_utils.create_df_trials/events`) |
| bonsai S3 (local Han NWB) | **legacy reader** |
| bpod S3 (local Han NWB) | **legacy reader** (+ h5py fallback for old files) |

The AIND reader runs **only on CO assets**. On **any** error (quality assertion, parse error,
empty/invalid/missing S3 location), it falls back to the **legacy reader on the first NWB that
reads**: the **local Han NWB** if present, then the **CO asset's own NWB** (a lenient read that
skips the AIND assertions) — logging the reason + kind. Only genuinely unreadable assets (NWB
missing at its S3 location, or an NWB with no trials table) hard-fail (logged).

### 5. CO NWB S3 path by construction, not glob
The NWB lives at a deterministic `{location}/nwb/{session_name}.nwb`, so we build that string
instead of a recursive `{location}/**/*.nwb` S3 glob — one S3 call instead of a full listing.

### 6. Coalescing (default; `--no-coalesce` to disable)
After the parallel per-session writes, each subject's files are merged into one sorted
`subject_id=<id>/<subject_id>.parquet`.
- Single-session queries stay **one S3 GET** (partition pruning still isolates the subject).
- Full-dataset loads open **~26× fewer files** (≈900 vs ≈23k) → much faster.
- **Query syntax is unchanged** — the standard glob + `hive_partitioning` + `session_id`-column
  filter works identically; only the on-disk file count differs.

### 7. Parallelism
- **Processes, oversubscribed** (`ProcessPoolExecutor`, `--n-workers`, default `CO_CPUS-1`).
  CO-asset reads are **I/O-bound** (S3 zarr), so going past the core count overlaps S3 latency:
  on a 16-core box **64 workers ≈ 4×** faster than default; beyond ~64 there's no gain (the
  `create_df_trials` parse saturates the cores). RAM is not the limit (~21 GB at 128 workers).
- **docDB enrichment stays at 20 threads** — docDB 503s under higher concurrency, and it isn't
  the bottleneck.

### 8. Observability
- `processing_log.csv` — cumulative, one row per session: status / data_source / reader /
  n_trials / n_events / nwb_path / error.
- `co_skipped_sessions.csv` — the multi-session-per-day CO rows that were skipped, with reason.

### 9. Bad-Bowen guard
Sessions listed in `Bowen_IncompleteSessions-081225.csv` have unreliable bonsai data; flagged
`is_bad_bowen_session` and only trusted via their CO asset (never the local bonsai NWB).

### 10. Cross-reader schema normalization
Three readers (AIND, legacy bonsai, legacy bpod) emit slightly different trial schemas. On
write (`_write_session_parquet`) we normalize so the partitioned tables stay queryable:
- `subject_id` is always cast to string (matches the `subject_id=` partition dir type);
- `bpod_backup_*` columns (raw hardware backup, ~22 cols, only in the legacy bpod reader) are
  dropped;
- a canonical type map (`_CANONICAL_TRIAL_COL_TYPES` / `_CANONICAL_EVENT_COL_TYPES`) resolves
  cross-session type conflicts (highest-priority type wins: string > double > int64 > bool).

Queries then use `union_by_name=true` to merge any remaining per-reader column differences.

---

## `build_cache` CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir` | local scratch | output dir or `s3://…` prefix (prod = `s3://aind-scratch-data/aind-dynamic-foraging-cache`) |
| `--n-workers` | `CO_CPUS-1` | worker processes; **~64 recommended** for the I/O-bound CO reads |
| `--limit N` | all | build only a random N-session subset (quick test) |
| `--full-rebuild` | off | ignore `build_metadata.json` and reprocess every session |
| `--no-coalesce` | off | keep one parquet file per session instead of per subject |
| `--co-cache PATH` | — | dev: cache the ~137 s docDB discovery parquet (load if present, else fetch+save) |

Incremental by default: re-running only processes sessions not already in `build_metadata.json`.

---

## Querying (DuckDB)

Always use the three options below: partition pruning, schema merge across readers, and the
`subject_id` cast (DuckDB infers the `subject_id=…` dir name as BIGINT).

```python
import duckdb

TRIAL = f"read_parquet('{OUT}/trial_table/**/*.parquet', hive_partitioning=true, union_by_name=true)"

df = duckdb.sql(f"""
    WITH sel AS (
        SELECT _session_id, subject_id, session_date, task, foraging_eff
        FROM read_parquet('{OUT}/session_table.parquet')
        WHERE task LIKE '%Uncoupled%' AND foraging_eff > 0.8
    )
    SELECT s.subject_id, s.session_date, t.session_id,
           t.animal_response, t.earned_reward,
           t.reward_probabilityL, t.reward_probabilityR
    FROM {TRIAL} t
    JOIN sel s ON t.session_id = s._session_id
    WHERE CAST(t.subject_id AS VARCHAR) IN (SELECT subject_id FROM sel)
    ORDER BY s.subject_id, s.session_date
""").df()
```

`OUT` can be a local dir or an `s3://` prefix — DuckDB reads S3 natively. See
`query_examples.py` / `query_examples.ipynb` for runnable demos.

---

## Performance (measured on the full prod cache — ~23.6k sessions, 12.5M trials, S3)

- **Build (full, 64 workers):** ~1 h, dominated by ~15k CO-asset S3 reads; incremental
  re-runs only touch new/unprocessed sessions (cheap). docDB discovery ≈ 137 s (cache it for
  dev with `--co-cache`).
- **Loading trials into memory from S3** (coalesced, ~900 files, via DuckDB → pandas):
  - **5-column projection** (choice/reward/prob) → **~4 s, ~1.2 GB** — the normal analysis pattern.
  - full 103-column table → **~53 s, ~21 GB**.
  - `COUNT(*)` over the trial table → ~1 s.
- **Return-loop join** (filter sessions → pull all their trials + events: 1.27M trials +
  13.4M events) → **~44 s** directly over S3.

Memory scales with columns selected (projection ≈ 17× less RAM); coalescing keeps the
file-open overhead small even for the full-width load.

---

## Validation

The cache was validated end-to-end against the legacy NWB read paths and Han's master
session table. Scripts live in [`validate/`](validate/) (`validate_step1.py`,
`validate_step2.py`, `plot_validation.py`); artifacts write to `scratch/tmp/validation/`.

**Step 1 — data equivalence + speed.** Sampling per source (co_asset / bonsai / bpod), the
cache returns *exactly* the same trial data as a direct NWB read (**33/33 sessions exact
match**: 5-col, full, and full+events). Fetch time is measured at every scale (median over
repeated, re-sampled draws) against the TRUE legacy CO chain run serially —
`get_subject_assets` (docDB query) → `add_s3_location` (S3 glob) → `nwb_utils.create_df_*`
— which is extrapolated (~23 s/session, dominated by the ~17 s docDB query):

| fetch | cache, full DB (~23.6k sessions) | legacy CO chain, full DB |
|---|---|---|
| 5-column trials | **~3 s** | ~23 s/session → **~6 days** |
| full 103-col trials | **~53 s** | ~23 s/session → ~6 days |
| trials + events | ~64 s (at 10k) | ~27 s/session |

→ the cache is **~4 orders of magnitude faster**; it eliminates the per-session docDB query
that dominates the legacy route.

![Cache vs legacy fetch time](validate/cache_vs_legacy.png)

**Step 2 — apples-to-apples vs Han's master table.** Han's session stats are specific *sums
over `df_trial`* (`process_nwbs.py`), and crucially Han's `total_trials` **excludes autowater
trials**. Reproducing each Han definition from the cache trial table
(`non_aw = auto_waterL==0 & auto_waterR==0`; `IGNORE=2`; left/right recovered from
`bias_naive`+`finished_trials`), all metrics agree per-session:

| metric | exact match | metric | exact match |
|---|---|---|---|
| total_trials (incl. autowater) | 97.4% | autowater_trials | 99.7% |
| total_trials (foraging, non-AW) | 97.4% | reward_trials (earned) | 98.0% |
| finished_trials | 97.8% | left / right choices | 98.0% |

By source: **bpod 100%, bonsai 99.8%** (same NWBs as Han → true apples-to-apples);
**co_asset 96.2%** is the only notable residual — those sessions read the AIND CO NWB while
Han read the bonsai NWB (different files/pipelines), plus ~14 truncated CO assets. Han's
pipeline trims no trials, so the cache reproduces Han's master table as a single source of
truth.

---

## Known limitations
- **Genuinely unreadable CO assets hard-fail** (~14 / 23.9k ≈ 0.06%, logged in
  `processing_log.csv` with `status=failed`): the NWB is missing at its S3 location, or the
  NWB has no trials table — no reader (AIND or legacy) can recover those.
- **Multi-session-per-day CO sessions** without a Han exact match are skipped, not merged —
  see `co_skipped_sessions.csv` and [issue #146](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils/issues/146).
- **~269 sessions are skipped** ("no NWB found"): in Han's table but with no local NWB and no
  CO asset (e.g. bad-Bowen without a CO asset, or NWB not mounted).
- Benign `s3fs`/`aiobotocore` "attached to a different loop" tracebacks during worker S3
  teardown are **not failures**; the `asyncio` logger is quieted in `build_cache` to keep logs
  readable. Trust the builder's `Failed to process` lines + `BUILD SUMMARY` + `processing_log.csv`.
- Build runs inside Code Ocean (local NWB mounts under `/data/`, `CO_CPUS` for worker count).
