"""
DuckDB query helpers for the foraging parquet cache.

Two layers — reach for the simple helpers first, drop to native SQL when you need more:

  Layer 1 (convenience — the common "return loop"):
      select_sessions -> fetch_trials / fetch_events
    Filter the (small) session table on any metric / metadata, then pull those sessions'
    trials or events with the session metadata already joined on — in one call.

  Layer 0 (escape hatch — covers ANY query):
      read_trials / read_events
    Return a fast, partition-scoped ``read_parquet(...)`` clause for a set of subjects.
    Drop it into whatever SQL you write — aggregations, window functions, trial<->event
    joins, custom GROUP BY. You keep the full power of SQL; the helper only does the part
    that is easy to get wrong or slow (reading the right partition files, fast + correct).

Why scoped reads are fast: a full ``trial_table/**/*.parquet`` glob with ``union_by_name``
must read *every* subject file's footer to build the column union before it can prune
(~25 s cold). Scoping the read to just the subjects you asked for reads only their footers
(~1 s), while still unioning their columns correctly.

Everything reads the public S3 cache (no AWS credentials needed). To query a local build,
pass ``base=`` (or reassign ``SESSION_DB`` / ``TRIAL_DB`` / ``EVENT_DB``).
"""

import duckdb

PROD_S3_PREFIX = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
SESSION_DB = f"{PROD_S3_PREFIX}/session_table.parquet"  # flat session table
TRIAL_DB = f"{PROD_S3_PREFIX}/trial_table"  # Hive-partitioned by subject_id
EVENT_DB = f"{PROD_S3_PREFIX}/event_table"  # Hive-partitioned by subject_id

# SELECT * over the trial table is ~21 GB — always project. These small defaults cover
# the usual choice/reward analysis; pass columns=[...] for others, or columns="*" for all.
DEFAULT_TRIAL_COLUMNS = [
    "trial", "animal_response", "earned_reward",
    "reward_probabilityL", "reward_probabilityR",
]
DEFAULT_EVENT_COLUMNS = ["trial", "timestamps", "event", "data"]

# Leading identity columns we always emit and never duplicate from the trial/event side.
_KEYS = ("subject_id", "session_date", "session_id")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _conn(con):
    """Return the DuckDB connection to use (the given one, or the default module conn)."""
    return con if con is not None else duckdb


def _quote_in(values):
    """Render an iterable as a SQL IN-list of quoted, escaped string literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _partition_subjects(base, con=None):
    """Subject ids that actually have a partition file under ``base``.

    One cheap S3 LIST via ``glob()`` (not a footer scan) — used to drop requested
    subjects with no files, since a scoped ``read_parquet`` list errors on a path that
    matches nothing.
    """
    rows = _conn(con).sql(f"SELECT file FROM glob('{base}/subject_id=*/*.parquet')").df()
    found = rows["file"].str.extract(r"subject_id=([^/]+)/")[0].dropna()
    return set(found)


def _full_glob(base):
    """The correct-but-slow read over every subject (reads all footers for the union)."""
    return f"read_parquet('{base}/**/*.parquet', hive_partitioning=true, union_by_name=true)"


def _scoped_read(base, subjects, con):
    """Build a ``read_parquet(...)`` clause scoped to ``subjects`` (or the full glob)."""
    if subjects is None:
        return _full_glob(base)
    want = {str(s) for s in subjects} & _partition_subjects(base, con)
    files = [f"'{base}/subject_id={s}/*.parquet'" for s in sorted(want)]
    if not files:
        # No requested subject has data: yield zero rows but the correct full schema.
        return f"(SELECT * FROM {_full_glob(base)} WHERE false)"
    return f"read_parquet([{', '.join(files)}], hive_partitioning=true, union_by_name=true)"


# ---------------------------------------------------------------------------
# Layer 0 — escape hatch: a fast, partition-scoped read_parquet(...) source
# ---------------------------------------------------------------------------


def read_trials(subjects=None, base=None, con=None):
    """Return a ``read_parquet(...)`` clause for the trial table, scoped to ``subjects``.

    Drop the returned string into any SQL you write::

        src = read_trials(['754372', '758435'])
        duckdb.sql(f"SELECT subject_id, AVG(earned_reward::DOUBLE) FROM {src} GROUP BY subject_id")

    Scoping to the subjects you need reads only their partition files (~1 s) instead of
    every subject's footer. ``subjects=None`` falls back to the full (slow) glob over all
    subjects. Note a scoped read exposes only the columns present in *those* subjects'
    files; selecting a column none of them has will raise.

    Parameters
    ----------
    subjects : iterable, optional
        Subject ids to scope the read to. ``None`` reads the full table (slow glob).
    base : str, optional
        Trial-table location — the partitioned-table **directory** prefix (default: the
        production S3 ``trial_table``). Pass a local dir / other S3 prefix for another build.
    con : duckdb connection, optional
        DuckDB connection to run the partition listing on (default: the module connection).
        Pass your own for warm reuse, or custom settings (S3 region/creds, threads, memory).
    """
    return _scoped_read(base or TRIAL_DB, subjects, con)


def read_events(subjects=None, base=None, con=None):
    """Return a ``read_parquet(...)`` clause for the event table, scoped to ``subjects``.

    The event-table counterpart of :func:`read_trials` — same ``subjects`` / ``base`` / ``con``
    behaviour, except ``base`` defaults to the production S3 ``event_table`` directory prefix.
    """
    return _scoped_read(base or EVENT_DB, subjects, con)


# ---------------------------------------------------------------------------
# Layer 1 — convenience: filter sessions, then fetch their trials / events
# ---------------------------------------------------------------------------


def select_sessions(where=None, subjects=None, columns=None, base=None, con=None,
                    order_by="subject_id, session_date"):
    """Filter the (small) session table; return the selected sessions as a DataFrame.

    The first step of both common workflows — filter on session metrics/metadata, or on
    subject first, or both — then hand the result to :func:`fetch_trials` /
    :func:`fetch_events`.

    Parameters
    ----------
    where : str, optional
        Raw SQL predicate on the session table, e.g.
        ``"task LIKE '%Uncoupled%' AND foraging_eff > 0.8"``.
    subjects : iterable, optional
        Restrict to these subject ids (adds ``subject_id IN (...)``).
    columns : list[str], optional
        Extra session-metadata columns to carry along (and onto trials/events later).
        ``_session_id, subject_id, session_date`` are always included as leading columns.
    base : str, optional
        Session table to read — the ``session_table.parquet`` **file** (default: the
        production S3 cache). Pass a local file / other S3 path to query another build.
    con : duckdb connection, optional
        DuckDB connection to run on (default: the module connection). Pass your own for warm
        reuse across calls, or custom settings (S3 region/creds, threads, memory).
    order_by : str, optional
        SQL ORDER BY clause (default: ``"subject_id, session_date"``); pass ``None`` for none.

    Returns
    -------
    pandas.DataFrame
        One row per selected session, with ``_session_id`` as the join key.
    """
    base = base or SESSION_DB
    extra = [c for c in (columns or []) if c not in ("_session_id", *_KEYS)]
    sel_cols = ", ".join(["_session_id", "subject_id", "session_date", *extra])
    clauses = []
    if subjects is not None:
        clauses.append(f"subject_id IN ({_quote_in(subjects)})")
    if where:
        clauses.append(f"({where})")
    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_sql = f"ORDER BY {order_by}" if order_by else ""
    return _conn(con).sql(
        f"SELECT {sel_cols} FROM read_parquet('{base}') {where_sql} {order_sql}"
    ).df()


def fetch_trials(sessions, columns=None, base=None, con=None):
    """Pull trial rows for a set of selected sessions, with session metadata joined on.

    Reads only the selected subjects' partitions (fast) and inner-joins to ``sessions`` on
    the session key, so exactly the selected sessions' trials are returned — each row
    carrying its session metadata.

    Parameters
    ----------
    sessions : pandas.DataFrame
        Selected sessions (e.g. from :func:`select_sessions`). Must contain ``_session_id``
        and ``subject_id``; every other column is carried onto each trial row.
    columns : list[str] or "*", optional
        Trial columns to project (default: a small choice/reward set). ``"*"`` returns all
        103 columns (large). Columns absent for the selected subjects come back all-NULL.
    base : str, optional
        Trial-table **directory** prefix (default: the production S3 ``trial_table``). Pass a
        local dir / other S3 prefix to query another build.
    con : duckdb connection, optional
        DuckDB connection to run on (default: the module connection). Pass your own for warm
        reuse across calls, or custom settings (S3 region/creds, threads, memory).

    Returns
    -------
    pandas.DataFrame
        One row per trial, leading ``subject_id, session_date, session_id``, ordered by
        ``subject_id, session_date, trial``.
    """
    return _fetch(sessions, base or TRIAL_DB, columns or DEFAULT_TRIAL_COLUMNS,
                  con, order_tail="trial")


def fetch_events(sessions, events=None, columns=None, base=None, con=None):
    """Pull event rows for a set of selected sessions, with session metadata joined on.

    Like :func:`fetch_trials`, for the event table.

    Parameters
    ----------
    sessions : pandas.DataFrame
        Selected sessions (needs ``_session_id`` and ``subject_id``).
    events : iterable, optional
        Restrict to these event types, e.g. ``['left_lick_time', 'right_lick_time']``.
    columns : list[str] or "*", optional
        Event columns to project (default: ``trial, timestamps, event, data``). Columns absent
        for the selected subjects come back all-NULL.
    base : str, optional
        Event-table **directory** prefix (default: the production S3 ``event_table``). Pass a
        local dir / other S3 prefix to query another build.
    con : duckdb connection, optional
        DuckDB connection to run on (default: the module connection). Pass your own for warm
        reuse across calls, or custom settings (S3 region/creds, threads, memory).

    Returns
    -------
    pandas.DataFrame
        One row per event, leading ``subject_id, session_date, session_id``, ordered by
        ``subject_id, session_date, timestamps``.
    """
    extra_where = f"t.event IN ({_quote_in(events)})" if events else None
    return _fetch(sessions, base or EVENT_DB, columns or DEFAULT_EVENT_COLUMNS,
                  con, order_tail="timestamps", extra_where=extra_where)


def _fetch(sessions, base, columns, con, order_tail, extra_where=None):
    """Shared core for fetch_trials / fetch_events: scoped read + join to selected sessions.

    A scoped read exposes only the columns present in *those* subjects' files (some columns
    are reader-specific, e.g. ``trial`` is absent from some legacy files). So we adapt to the
    columns actually available: requested columns that are missing are emitted as all-NULL
    (stable output shape, never an error), and the ORDER BY tail is dropped if absent.
    """
    import pandas as pd

    if len(sessions) == 0:
        return pd.DataFrame()
    conn = _conn(con)
    src = _scoped_read(base, sessions["subject_id"].unique().tolist(), con)
    avail = set(conn.sql(f"DESCRIBE SELECT * FROM {src}").df()["column_name"])
    conn.register("_sel_sessions", sessions)

    meta = [f"s.{c}" for c in sessions.columns if c not in ("_session_id", *_KEYS)]
    if columns in ("*", ["*"]):
        proj = [f"t.* EXCLUDE ({', '.join(k for k in _KEYS if k in avail)})"]
    else:
        proj = [f"t.{c}" if c in avail else f"CAST(NULL AS DOUBLE) AS {c}"
                for c in columns if c not in _KEYS]
    select = ", ".join(["s.subject_id", "s.session_date", "t.session_id", *meta, *proj])
    where_sql = f"WHERE {extra_where}" if extra_where else ""
    order = ["s.subject_id", "s.session_date"]
    if order_tail in avail:
        order.append(f"t.{order_tail}")
    try:
        return conn.sql(f"""
            SELECT {select}
            FROM {src} t
            JOIN _sel_sessions s ON t.session_id = s._session_id
            {where_sql}
            ORDER BY {', '.join(order)}
        """).df()
    finally:
        conn.unregister("_sel_sessions")
