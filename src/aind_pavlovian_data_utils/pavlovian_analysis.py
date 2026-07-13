"""
Auto-detecting Pavlovian fiber-photometry analysis and visualization.

Builds on this package's existing utilities rather than re-reading raw NWB:
    nwb_utils.load_nwb_from_filename / create_df_events / create_df_fip
    alignment.event_triggered_response  (PSTH / ETR engine)

Public functions:
    canonical_event_name
    load_pavlovian_dfs
    detect_paradigm
    classify_trials
    compute_psth
    plot_session_overview
    plot_cs_psth_grid
    plot_lick_quant
    plot_reaction_time
    analyze_nwb

The paradigm (how many CS, reward-only vs reward+airpuff) is inferred from the
events table, so the same entry point handles Stage0-3 (and "Custom"):
    Stage1 : 1CS-US (reward)
    Stage2 : 3CS-US (reward)
    Stage3 : 4CS-US (reward / airpuff)
    Stage0 : 2CS-US (reward / airpuff)

Visualization covers ALL available channels (Iso / Green / Red) and ALL ROIs.
"""

import re
import warnings

import matplotlib

matplotlib.use("Agg")  # capsule/headless safe; callers may override before import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from . import nwb_utils  # noqa: E402
from .alignment import event_triggered_response  # noqa: E402

# NWB channel prefix -> display label
CHANNEL_MAP = {"Iso": "Iso", "G": "Green", "R": "Red"}
CHANNEL_ORDER = ["Iso", "Green", "Red"]
CHANNEL_COLOR = {"Iso": "blue", "Green": "green", "Red": "magenta"}

DEFAULT_PREPROCESSING = "dff-bright_mc-iso-IRLS"

# CS -> (US kind, plot color, US-delivered label, omission label)
CS_INFO = {
    "CS1": ("reward", (1.0, 0.0, 0.0), "R+", "R-"),
    "CS2": ("reward", (0.0, 0.7, 0.0), "R+", "R-"),
    "CS3": ("reward", (1.0, 0.0, 1.0), "R+", "R-"),
    "CS4": ("airpuff", (0.3, 0.3, 0.3), "P+", "P-"),
}

# default PSTH window (seconds, relative to CS onset)
T_BEFORE = 5.0
T_AFTER = 15.0
BASELINE = 5.0
OUTPUT_SR = 20.0  # Hz, ETR interpolation grid


def canonical_event_name(raw):
    """Map a raw pavlovian ``events`` string to a canonical key.

    Returns one of ``Lick``/``Reward``/``Airpuff``/``CS1``..``CS4`` or ``None``.
    """
    s = str(raw).strip().lower()
    if "lick" in s:
        return "Lick"
    if "airpuff" in s or "air_puff" in s or "puff" in s:
        return "Airpuff"
    if "reward" in s or "water" in s:
        return "Reward"
    m = re.search(r"cs[_\s]*([1-4])", s)
    if m:
        return "CS" + m.group(1)
    return None


def _pick_adjust_time(nwb):
    """Return True if the session has a ``CS_start_time`` column to align to.

    ``create_df_events`` / ``create_df_fip`` subtract the first ``CS_start_time``
    when ``adjust_time=True``; if that column is absent we must use absolute time.
    """
    try:
        trials = nwb.trials
        return "CS_start_time" in set(trials.colnames)
    except Exception:
        return False


def load_pavlovian_dfs(nwb_or_path, preprocessing=DEFAULT_PREPROCESSING, adjust_time=None):
    """Load and prepare the events / FIP dataframes for one session.

    Uses ``nwb_utils.create_df_events`` and ``create_df_fip`` (so timestamps are
    seconds on a shared clock). The FIP frame is filtered to the requested
    ``preprocessing`` variant and annotated with ``channel`` (Iso/Green/Red) and
    ``roi`` (int) columns parsed from the TimeSeries name.

    Returns
    -------
    df_events : pd.DataFrame
        Columns include ``timestamp`` (s), ``events``, ``trial``, ``canonical``.
    df_fip : pd.DataFrame
        Tidy FIP for the selected variant with added ``channel`` / ``roi``.
    meta : dict
        ``subject_id``, ``date``, ``sampling_rate`` (s), ``adjust_time``.
    """
    nwb = nwb_utils.load_nwb_from_filename(nwb_or_path)
    if adjust_time is None:
        adjust_time = _pick_adjust_time(nwb)

    df_events = nwb_utils.create_df_events(nwb, adjust_time=adjust_time, verbose=False)
    df_events = df_events.copy()
    df_events["canonical"] = df_events["events"].map(canonical_event_name)

    df_fip = nwb_utils.create_df_fip(nwb, tidy=True, adjust_time=adjust_time, verbose=False)
    if df_fip is None or len(df_fip) == 0:
        raise ValueError("No fiber-photometry data returned by create_df_fip.")

    suffix = "_" + preprocessing
    sel = df_fip[df_fip["event"].astype(str).str.endswith(suffix)].copy()
    if len(sel) == 0:
        available = sorted(df_fip["event"].astype(str).unique())[:6]
        raise ValueError(
            "No FIP signals for preprocessing '%s'. Available e.g.: %s" % (preprocessing, available)
        )

    def _parse(name):
        """Parse '<Chan>_<ROI>_<preprocessing>' -> (channel_label, roi_int) or (None, None)."""
        base = name[: -len(suffix)]
        m = re.match(r"(Iso|G|R)_(\d+)$", base)
        if not m:
            return None, None
        return CHANNEL_MAP[m.group(1)], int(m.group(2))

    parsed = sel["event"].astype(str).map(_parse)
    sel["channel"] = [p[0] for p in parsed]
    sel["roi"] = [p[1] for p in parsed]
    sel = sel[sel["channel"].notna()].copy()

    # effective sampling rate from the FIP timestamps
    ts = np.sort(sel["timestamps"].unique())
    sr = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else OUTPUT_SR

    # subject / date best-effort from session_id
    sid = str(getattr(nwb, "session_id", "") or "")
    subject_id, date = "unknown", "unknown"
    if sid.startswith("behavior") or sid.startswith("FIP"):
        parts = sid.split("_")
        if len(parts) >= 3:
            subject_id, date = parts[1], parts[2]
    else:
        parts = sid.split("_")
        if len(parts) >= 2:
            subject_id, date = parts[0], parts[1].replace(".json", "")
    subj = getattr(nwb, "subject", None)
    subject_id = (getattr(subj, "subject_id", "") if subj else "") or subject_id

    meta = {
        "subject_id": subject_id,
        "date": date,
        "sampling_rate": sr,
        "adjust_time": bool(adjust_time),
    }
    return df_events, sel, meta


def detect_paradigm(df_events):
    """Infer the paradigm from the canonical events present.

    Returns a dict with ``stage`` (str), ``cs_list`` (list of CS names present),
    and ``has_airpuff`` (bool).
    """
    present = set(df_events["canonical"].dropna().unique())
    active = [c for c in ("CS1", "CS2", "CS3", "CS4") if c in present]
    has_airpuff = "Airpuff" in present

    cs_list = [c for c in active if not (CS_INFO[c][0] == "airpuff" and not has_airpuff)]
    stage = {
        (1, False): "Stage1 (1CS-US, reward)",
        (3, False): "Stage2 (3CS-US, reward)",
        (2, True): "Stage0 (2CS-US, reward/airpuff)",
        (4, True): "Stage3 (4CS-US, reward/airpuff)",
    }.get(
        (len(cs_list), has_airpuff),
        "Custom (%dCS-US%s)" % (len(cs_list), ", airpuff" if has_airpuff else ""),
    )
    return {"stage": stage, "cs_list": cs_list, "has_airpuff": has_airpuff}


def classify_trials(df_events, cs_list):
    """Split each CS's trials into US-delivered (pos) vs omission (neg).

    Uses the events table ``trial`` column: a CS trial is positive when that
    trial number also contains the matching US event (reward or airpuff).

    Returns ``{cs_name: {"onsets": np.ndarray, "pos_mask": np.ndarray(bool),
    "trials": np.ndarray}}`` where ``onsets`` are CS-onset timestamps (s).
    """
    reward_trials = set(df_events.loc[df_events["canonical"] == "Reward", "trial"].astype(int))
    airpuff_trials = set(df_events.loc[df_events["canonical"] == "Airpuff", "trial"].astype(int))
    us_by_kind = {"reward": reward_trials, "airpuff": airpuff_trials}

    out = {}
    for cs in cs_list:
        rows = df_events[df_events["canonical"] == cs].sort_values("timestamp")
        onsets = rows["timestamp"].to_numpy(float)
        trials = rows["trial"].astype(int).to_numpy()
        delivered = us_by_kind[CS_INFO[cs][0]]
        pos_mask = np.array([t in delivered for t in trials], dtype=bool)
        out[cs] = {"onsets": onsets, "pos_mask": pos_mask, "trials": trials}
    return out


def compute_psth(
    df_fip,
    channel,
    roi,
    event_times,
    t_before=T_BEFORE,
    t_after=T_AFTER,
    baseline=BASELINE,
    output_sampling_rate=OUTPUT_SR,
):
    """Event-triggered response for one channel/ROI, baseline-subtracted.

    Wraps ``alignment.event_triggered_response`` and returns
    ``(time, mean, sem, n)`` in percent dF/F. Empty inputs yield zero-length
    ``mean``/``sem`` and ``n == 0``.
    """
    sub = df_fip[(df_fip["channel"] == channel) & (df_fip["roi"] == roi)]
    if len(sub) == 0 or len(event_times) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    data = sub[["timestamps", "data"]].sort_values("timestamps").reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        etr = event_triggered_response(
            data=data,
            t="timestamps",
            y="data",
            event_times=list(event_times),
            t_start=-abs(t_before),
            t_end=abs(t_after),
            output_sampling_rate=output_sampling_rate,
            output_format="tidy",
            interpolate=True,
            censor=False,
            nan_policy="interpolate",
        )
    if etr is None or len(etr) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    # tidy -> matrix [time, event]
    wide = etr.pivot_table(index="time", columns="event_number", values="data")
    t = wide.index.to_numpy(float)
    mat = wide.to_numpy(float) * 100.0  # percent dF/F
    base = t < (t[0] + baseline)
    if base.any():
        mat = mat - np.nanmean(mat[base, :], axis=0, keepdims=True)
    mean = np.nanmean(mat, axis=1)
    n = mat.shape[1]
    sem = np.nanstd(mat, axis=1) / np.sqrt(max(n, 1))
    return t, mean, sem, n


def _channels_present(df_fip, channels=None):
    """Return ordered (channel_label, roi) pairs present, honoring a filter.

    ``channels`` is an optional dict keyed by ``'<Chan>_<ROI>'`` base names.
    """
    if channels:
        wanted = set()
        for key in channels:
            m = re.match(r"(Iso|G|R)_(\d+)$", str(key))
            if m:
                wanted.add((CHANNEL_MAP[m.group(1)], int(m.group(2))))
        pairs = [
            (c, r)
            for (c, r) in sorted(
                set(zip(df_fip["channel"], df_fip["roi"])),
                key=lambda cr: (CHANNEL_ORDER.index(cr[0]), cr[1]),
            )
            if (c, r) in wanted
        ]
    else:
        pairs = sorted(
            set(zip(df_fip["channel"], df_fip["roi"])),
            key=lambda cr: (CHANNEL_ORDER.index(cr[0]), cr[1]),
        )
    return pairs


def plot_session_overview(df_events, df_fip, paradigm, meta, channels=None):
    """Whole-session traces for every channel/ROI with CS/US/lick markers."""
    pairs = _channels_present(df_fip, channels)
    rois = sorted({r for _, r in pairs})
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]

    fig, ax = plt.subplots(figsize=(20, 3 + 1.2 * max(len(rois), 1)))
    for i, roi in enumerate(rois):
        off = -i * 100
        for c in chans:
            sub = df_fip[(df_fip["channel"] == c) & (df_fip["roi"] == roi)]
            sub = sub.sort_values("timestamps")
            if len(sub):
                ax.plot(sub["timestamps"], sub["data"] * 100 + off, color=CHANNEL_COLOR[c], lw=0.6)
        ax.axhline(off, ls="--", color="k", lw=0.5)
        ax.text(df_fip["timestamps"].max(), off, "  ROI%d" % roi, va="center", fontsize=8)

    def _mark(key, color, width, label):
        """Shade spans for a canonical event key and add one legend proxy."""
        times = df_events.loc[df_events["canonical"] == key, "timestamp"].to_numpy(float)
        for tt in times:
            ax.axvspan(tt, tt + width, color=color)
        if len(times):
            ax.axvspan(0, 0, color=color, label=label)

    _mark("Reward", (0, 0, 1, 0.35), 0.5, "Reward")
    _mark("Airpuff", (0, 0, 0, 0.35), 0.5, "Airpuff")
    for cs in paradigm["cs_list"]:
        _mark(cs, CS_INFO[cs][1] + (0.30,), 1.0, cs)

    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float)
    if len(licks):
        ax.plot(
            licks,
            np.full(len(licks), 100),
            marker=3,
            ms=6,
            ls="none",
            color=(0, 0, 0, 0.5),
            label="Lick",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F (%)  (ROIs offset)")
    ax.set_title("%s  %s   [%s]" % (meta["subject_id"], meta["date"], paradigm["stage"]))
    ax.grid(True)
    ax.legend(loc="upper right", ncol=6, fontsize=8)
    fig.tight_layout()
    return fig


def plot_cs_psth_grid(
    df_fip,
    cs,
    cls,
    meta,
    channels=None,
    t_before=T_BEFORE,
    t_after=T_AFTER,
    baseline=BASELINE,
    output_sampling_rate=OUTPUT_SR,
):
    """PSTH grid for one CS: rows = ROI, cols = channel, pos vs neg overlaid."""
    pairs = _channels_present(df_fip, channels)
    rois = sorted({r for _, r in pairs})
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    us_kind, cs_color, pos_lab, neg_lab = CS_INFO[cs]

    onsets = cls[cs]["onsets"]
    pos_mask = cls[cs]["pos_mask"]
    pos_t, neg_t = onsets[pos_mask], onsets[~pos_mask]

    fig, axes = plt.subplots(
        len(rois), len(chans), figsize=(4.5 * len(chans), 3 * max(len(rois), 1)), squeeze=False
    )
    fig.suptitle(
        "%s %s  —  %s PSTH (%s vs %s)  n+=%d n-=%d"
        % (
            meta["subject_id"],
            meta["date"],
            cs,
            pos_lab,
            neg_lab,
            int(pos_mask.sum()),
            int((~pos_mask).sum()),
        ),
        fontsize=13,
    )

    for ci, c in enumerate(chans):
        for ri, roi in enumerate(rois):
            ax = axes[ri][ci]
            for times, main, sub, lab in (
                (pos_t, cs_color, cs_color, pos_lab),
                (neg_t, "k", "gray", neg_lab),
            ):
                t, mean, sem, n = compute_psth(
                    df_fip, c, roi, times, t_before, t_after, baseline, output_sampling_rate
                )
                if n > 0:
                    ax.plot(t, mean, color=main, label="%s (n=%d)" % (lab, n))
                    ax.fill_between(t, mean - sem, mean + sem, color=sub, alpha=0.35)
            ax.axvspan(0, 1, color=cs_color + (0.20,))
            ax.axhline(0, color="gray", ls="--", lw=0.6)
            ax.set_xlim(-t_before, t_after)
            ax.grid(True)
            if ri == 0:
                ax.set_title(c)
            if ci == 0:
                ax.set_ylabel("ROI%d\ndF/F (%%)" % roi)
            if ri == len(rois) - 1:
                ax.set_xlabel("Time - CS (s)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_lick_quant(df_events, paradigm, cls):
    """Anticipatory vs consummatory lick counts per trial, per CS."""
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float)
    if len(licks) == 0 or len(paradigm["cs_list"]) == 0:
        return None
    cs_list = paradigm["cs_list"]
    fig, axes = plt.subplots(1, len(cs_list), figsize=(5 * len(cs_list), 4), squeeze=False)
    for j, cs in enumerate(cs_list):
        ax = axes[0][j]
        onsets = cls[cs]["onsets"]
        pos_mask = cls[cs]["pos_mask"]
        anti = np.array([np.sum((licks > o) & (licks < o + 2.0)) for o in onsets])
        post = np.array([np.sum((licks > o + 2.0) & (licks < o + 7.0)) for o in onsets])
        ax.plot(anti, label="Anticipatory")
        ax.plot(post, label="Consummatory/Omission")
        idx = np.arange(len(onsets))
        if pos_mask.any():
            ax.plot(idx[pos_mask], post[pos_mask], ".", color="blue", ms=9, label=CS_INFO[cs][2])
        if (~pos_mask).any():
            ax.plot(idx[~pos_mask], post[~pos_mask], ".", color="red", ms=9, label=CS_INFO[cs][3])
        mean_anti = float(np.mean(anti)) if len(anti) else float("nan")
        ax.set_title("%s  antiLick %.2f" % (cs, mean_anti))
        ax.set_xlabel("trial #")
        if j == 0:
            ax.set_ylabel("Lick #")
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_reaction_time(df_events):
    """Reaction time from each reward to the next lick (seconds)."""
    rew = df_events.loc[df_events["canonical"] == "Reward", "timestamp"].to_numpy(float)
    lick = np.sort(df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float))
    if len(rew) == 0 or len(lick) == 0:
        return None
    rt = np.full(len(rew), np.nan)
    for i, r in enumerate(rew):
        nxt = lick[np.searchsorted(lick, r, side="left") :]
        if len(nxt):
            rt[i] = nxt[0] - r
    good = rt[~np.isnan(rt)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(rt)
    ax[0].set_xlabel("Reward #")
    ax[0].set_ylabel("RT (s)")
    ax[0].set_title("median RT %.3f s" % (np.nanmedian(rt) if len(good) else float("nan")))
    if len(good):
        ax[1].hist(good)
        ax[2].hist(good[good < 0.5])
    ax[1].set_xlabel("RT (s)")
    ax[2].set_xlabel("RT < 0.5 s")
    fig.tight_layout()
    return fig


def _resolve_want(plot_types):
    """Resolve the requested plot set into a concrete subset of names."""
    if plot_types is None or "all" in plot_types or "all_sess" in plot_types:
        return {"session", "psth", "lick", "rt"}
    return set(plot_types)


def _make_figures(df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw):
    """Build the requested figures and return them as a list."""
    figs = []
    if "session" in want:
        figs.append(plot_session_overview(df_events, df_fip, paradigm, meta, channels))
    if "psth" in want:
        for cs in paradigm["cs_list"]:
            figs.append(plot_cs_psth_grid(df_fip, cs, cls, meta, channels, **psth_kw))
    if "lick" in want:
        f = plot_lick_quant(df_events, paradigm, cls)
        if f is not None:
            figs.append(f)
    if "rt" in want:
        f = plot_reaction_time(df_events)
        if f is not None:
            figs.append(f)
    return figs


def _build_summary(paradigm, cls, meta, chan_labels, n_roi):
    """Assemble (and print) the numeric summary dict for a session."""
    summary = {
        "subject_id": meta["subject_id"],
        "date": meta["date"],
        "stage": paradigm["stage"],
        "n_roi": n_roi,
        "channels": chan_labels,
        "sampling_rate": meta["sampling_rate"],
        "adjust_time": meta["adjust_time"],
        "cs": {},
    }
    print(
        "[detected] %s | CS=%s | ROIs=%d | channels=%s | sr=%.2fHz | adjust_time=%s"
        % (
            paradigm["stage"],
            paradigm["cs_list"],
            n_roi,
            chan_labels,
            meta["sampling_rate"],
            meta["adjust_time"],
        )
    )
    for cs in paradigm["cs_list"]:
        pm = cls[cs]["pos_mask"]
        summary["cs"][cs] = {
            "n_total": int(len(pm)),
            "n_pos": int(pm.sum()),
            "n_neg": int((~pm).sum()),
        }
        print(
            "  %s: %d trials (%d %s / %d %s)"
            % (cs, len(pm), int(pm.sum()), CS_INFO[cs][2], int((~pm).sum()), CS_INFO[cs][3])
        )
    return summary


def analyze_nwb(
    nwb_or_path,
    preprocessing=DEFAULT_PREPROCESSING,
    channels=None,
    save_path=None,
    plot_types=None,
    adjust_time=None,
    t_before=T_BEFORE,
    t_after=T_AFTER,
    baseline=BASELINE,
    output_sampling_rate=OUTPUT_SR,
):
    """End-to-end: load -> detect -> classify -> visualize a Pavlovian session.

    Parameters
    ----------
    nwb_or_path : str or NWBFile
        A combined behavior+fiber NWB (path or object).
    preprocessing : str
        dF/F variant suffix, e.g. ``'dff-bright_mc-iso-IRLS'``.
    channels : dict or None
        Optional ``{'<Chan>_<ROI>': 'location'}`` filter; None -> all present.
    save_path : str or None
        If given, a multi-page summary PDF is written there.
    plot_types : list or None
        ``['all_sess']``/``['all']`` -> everything, or a subset of
        ``{'session','psth','lick','rt'}``.

    Returns
    -------
    dict
        Summary with stage, per-CS trial counts, channels, and roi count.
    """
    df_events, df_fip, meta = load_pavlovian_dfs(nwb_or_path, preprocessing, adjust_time)
    paradigm = detect_paradigm(df_events)
    cls = classify_trials(df_events, paradigm["cs_list"])

    want = _resolve_want(plot_types)
    pairs = _channels_present(df_fip, channels)
    chan_labels = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    n_roi = len({r for _, r in pairs})

    summary = _build_summary(paradigm, cls, meta, chan_labels, n_roi)

    psth_kw = {
        "t_before": t_before,
        "t_after": t_after,
        "baseline": baseline,
        "output_sampling_rate": output_sampling_rate,
    }
    figs = _make_figures(df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw)

    if save_path:
        with PdfPages(save_path) as pdf:
            for fig in figs:
                pdf.savefig(fig)
        summary["pdf"] = save_path
        print("[saved] %s" % save_path)

    for fig in figs:
        plt.close(fig)
    return summary
