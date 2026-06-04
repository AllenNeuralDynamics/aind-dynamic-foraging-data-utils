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

The AIND reader runs **only on CO assets**. On **any** error it raises and we **fall back to
the legacy reader on the local Han NWB** (if one exists), logging the reason + kind. CO-only
sessions with no local NWB and an unreadable asset are the only hard failures (logged).

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

## Performance (measured)

- **Build (full ~24k sessions, 64 workers):** ~1–1.5 h, dominated by CO-asset S3 reads;
  incremental re-runs are cheap. docDB discovery ≈ 137 s (cache it for dev with `--co-cache`).
- **Loading trials into memory** (coalesced; full DB ≈ 12.5M trials):
  - **column projection** (e.g. choice/reward/prob) → **~1.2 GB**, tens of seconds — the
    normal analysis pattern.
  - full ~103-column table → ~21 GB, ~1–2 min.
  Memory scales with columns selected; coalescing keeps the file-open overhead small.

---

## Known limitations
- **CO-only sessions that the AIND reader can't read** have no Han NWB to fall back to → they
  hard-fail (logged in `processing_log.csv`). Rare (~0.2%).
- **Multi-session-per-day CO sessions** without a Han exact match are skipped, not merged —
  see `co_skipped_sessions.csv` and [issue #146](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils/issues/146).
- Build runs inside Code Ocean (local NWB mounts under `/data/`, `CO_CPUS` for worker count).
