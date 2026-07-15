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

from . import nwb_utils  # noqa: E402
from .alignment import event_triggered_response  # noqa: E402

# Data-shaping constants live in nwb_utils; re-used here for viz.
canonical_event_name = nwb_utils.canonical_event_name  # re-export for convenience
CHANNEL_MAP = nwb_utils.FIP_CHANNEL_MAP  # {'Iso':'Iso','G':'Green','R':'Red'}
DEFAULT_PREPROCESSING = nwb_utils.DEFAULT_FIP_PREPROCESSING

CHANNEL_ORDER = ["Iso", "Green", "Red"]
CHANNEL_COLOR = {"Iso": "blue", "Green": "green", "Red": "magenta"}

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
# ETR resamples onto this grid by interpolation; it need not equal the FIP
# acquisition rate, it only sets the output resolution (FIP is ~20 Hz).
OUTPUT_SR = 20.0
PNG_DPI = 150  # resolution of the PNG export

# anticipatory-lick summary defaults
ANTILICK_WINDOW = (0.0, 2.0)  # seconds after CS onset (CS->US delay)
ANTILICK_SUMMARY = "mean"  # 'mean' -> mean +/- SD, 'median' -> median +/- IQR
ANTILICK_SWARM_WIDTH = 0.28


def load_pavlovian_dfs(nwb_or_path, preprocessing=DEFAULT_PREPROCESSING, adjust_time=None):
    """Load the analysis-ready events / FIP dataframes for one session.

    Thin orchestration over ``nwb_utils`` (which does all the NWB manipulation:
    ms->s conversion, ``adjust_time`` alignment, canonical event labels, and the
    ``preprocessing`` filter with ``channel``/``roi`` parsing).

    Parameters
    ----------
    nwb_or_path : str or NWBFile
        Session to read.
    preprocessing : str
        dF/F variant suffix, e.g. ``'dff-bright_mc-iso-IRLS'``.
    adjust_time : bool or None
        Align time to the first CS. ``None`` -> auto (True when the session has a
        ``CS_start_time`` trials column, else False).

    Returns
    -------
    df_events : pandas.DataFrame
        Tidy events with ``timestamp`` (s), ``events``, ``trial``, ``canonical``.
    df_fip : pandas.DataFrame
        Tidy FIP for the selected variant with ``channel`` / ``roi``.
    meta : dict
        ``subject_id``, ``date``, ``adjust_time``.
    """
    nwb = nwb_utils.load_nwb_from_filename(nwb_or_path)
    if adjust_time is None:
        adjust_time = nwb_utils.can_align_to_cs(nwb)

    df_events = nwb_utils.create_df_events(nwb, adjust_time=adjust_time, verbose=False)
    df_fip = nwb_utils.create_df_fip_pavlovian(
        nwb, preprocessing=preprocessing, adjust_time=adjust_time, verbose=False
    )
    subject_id, session_date = nwb_utils.parse_session_name(nwb)
    meta = {"subject_id": subject_id, "date": session_date, "adjust_time": bool(adjust_time)}
    return df_events, df_fip, meta


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


def plot_session_overview(df_events, df_fip, paradigm, meta, channels=None, fig=None):
    """Whole-session traces for every channel/ROI with CS/US/lick markers.

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    pairs = _channels_present(df_fip, channels)
    rois = sorted({r for _, r in pairs})
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]

    if fig is None:
        fig = plt.figure(figsize=(20, 3 + 1.2 * max(len(rois), 1)))
    ax = fig.subplots()
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
    ax.set_title(
        "Session overview — %s %s [%s]" % (meta["subject_id"], meta["date"], paradigm["stage"]),
        fontsize=10,
    )
    ax.grid(True)
    ax.legend(loc="upper right", ncol=6, fontsize=8)
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
    fig=None,
):
    """PSTH grid for one CS: rows = ROI, cols = channel, pos vs neg overlaid.

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    pairs = _channels_present(df_fip, channels)
    rois = sorted({r for _, r in pairs})
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    us_kind, cs_color, pos_lab, neg_lab = CS_INFO[cs]

    onsets = cls[cs]["onsets"]
    pos_mask = cls[cs]["pos_mask"]
    pos_t, neg_t = onsets[pos_mask], onsets[~pos_mask]

    if fig is None:
        fig = plt.figure(figsize=(4.5 * len(chans), 3 * max(len(rois), 1)))
    axes = fig.subplots(len(rois), len(chans), squeeze=False)
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
    return fig


def plot_lick_quant(df_events, paradigm, cls, fig=None):
    """Anticipatory vs consummatory lick counts per trial, per CS.

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float)
    if len(licks) == 0 or len(paradigm["cs_list"]) == 0:
        return None
    cs_list = paradigm["cs_list"]
    if fig is None:
        fig = plt.figure(figsize=(5 * len(cs_list), 4))
    axes = fig.subplots(1, len(cs_list), squeeze=False)
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
    return fig


def _anticipatory_lick_counts(licks, onsets, window_s):
    """Per-trial lick count inside ``[onset+w0, onset+w1)`` for each CS onset."""
    w0, w1 = window_s
    return np.array([int(np.sum((licks >= o + w0) & (licks < o + w1))) for o in onsets], dtype=int)


def _beeswarm_offsets(values, width):
    """Symmetric x-offsets so equal (tied) values fan out instead of overlapping."""
    values = np.asarray(values, float)
    offsets = np.zeros(len(values))
    groups = {}
    for idx, v in enumerate(values):
        groups.setdefault(round(float(v)), []).append(idx)
    for idxs in groups.values():
        k = len(idxs)
        if k == 1:
            offsets[idxs[0]] = 0.0
        else:
            for pos, idx in zip(np.linspace(-width, width, k), idxs):
                offsets[idx] = pos
    return offsets


def _antilick_label(cs, cls):
    """X-axis label for a CS: name + empirical US-delivery rate (e.g. 'CS3\\n88%')."""
    pm = cls[cs]["pos_mask"]
    pct = (100.0 * pm.sum() / len(pm)) if len(pm) else 0.0
    return "%s\n%.0f%%" % (cs, pct)


def plot_anticipatory_lick_summary(
    df_events,
    paradigm,
    cls,
    meta,
    window_s=ANTILICK_WINDOW,
    summary=ANTILICK_SUMMARY,
    swarm_width=ANTILICK_SWARM_WIDTH,
    fig=None,
):
    """Beeswarm of anticipatory lick counts per CS (one dot = one trial).

    A ``mean +/- SD`` marker sits beside each cloud (``summary='median'`` ->
    ``median +/- IQR``). Draws into ``fig`` (a Figure or SubFigure) when given.
    """
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float)
    names = paradigm["cs_list"]
    if len(licks) == 0 or not names:
        return None

    if fig is None:
        fig = plt.figure(figsize=(max(4.2, 1.6 * len(names) + 2.0), 3.6))
    ax = fig.subplots()

    tick_pos, tick_lab, all_counts = [], [], []
    for i, cs in enumerate(names):
        counts = _anticipatory_lick_counts(licks, cls[cs]["onsets"], window_s)
        all_counts.append(counts)
        col = CS_INFO[cs][1]
        swarm_x, summ_x = i - 0.18, i + 0.22
        if len(counts):
            dx = _beeswarm_offsets(counts.astype(float), swarm_width)
            ax.scatter(
                swarm_x + dx,
                counts,
                s=26,
                facecolor=col,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.75,
                zorder=2,
            )
            if summary == "median":
                center = float(np.median(counts))
                lo_e = float(np.percentile(counts, 25))
                hi_e = float(np.percentile(counts, 75))
            else:  # mean +/- SD
                center = float(np.mean(counts))
                sd = float(np.std(counts))
                lo_e, hi_e = center - sd, center + sd
            ax.plot(
                [summ_x, summ_x],
                [lo_e, hi_e],
                color="black",
                lw=3.2,
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(summ_x, center, "o", color="black", ms=7, zorder=4)
        tick_pos.append(i)
        tick_lab.append(_antilick_label(cs, cls))

    ax.set_ylabel("Lick #")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ymax = max((c.max() for c in all_counts if len(c)), default=1)
    ax.set_ylim(-0.5, ymax * 1.08 + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.set_title(
        "%s  %s   anticipatory lick (%.0f-%.0fs)"
        % (meta["subject_id"], meta["date"], window_s[0], window_s[1]),
        fontsize=9,
    )
    return fig


def _anticipatory_summary(df_events, paradigm, cls, window_s):
    """Per-CS anticipatory-lick numbers for the JSON/return summary."""
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float)
    out = {}
    for cs in paradigm["cs_list"]:
        counts = _anticipatory_lick_counts(licks, cls[cs]["onsets"], window_s).astype(float)
        pm = cls[cs]["pos_mask"]
        rec = {
            "anti_window_s": [float(window_s[0]), float(window_s[1])],
            "anti_lick_mean": float(np.mean(counts)) if len(counts) else float("nan"),
            "anti_lick_sd": float(np.std(counts)) if len(counts) else float("nan"),
            "anti_lick_mean_pos": (
                float(np.mean(counts[pm])) if len(counts) and pm.any() else float("nan")
            ),
            "anti_lick_mean_neg": (
                float(np.mean(counts[~pm])) if len(counts) and (~pm).any() else float("nan")
            ),
        }
        out[cs] = rec
    return out


def plot_reaction_time(df_events, fig=None):
    """Reaction time from each reward to the next lick (seconds).

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
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
    if fig is None:
        fig = plt.figure(figsize=(12, 4))
    ax = fig.subplots(1, 3)
    ax[0].plot(rt)
    ax[0].set_xlabel("Reward #")
    ax[0].set_ylabel("RT (s)")
    ax[0].set_title("median RT %.3f s" % (np.nanmedian(rt) if len(good) else float("nan")))
    if len(good):
        ax[1].hist(good)
        ax[2].hist(good[good < 0.5])
    ax[1].set_xlabel("RT (s)")
    ax[2].set_xlabel("RT < 0.5 s")
    return fig


def _resolve_want(plot_types):
    """Resolve the requested plot set into a concrete subset of names."""
    if plot_types is None or "all" in plot_types or "all_sess" in plot_types:
        return {"session", "psth", "lick", "antilick", "rt"}
    return set(plot_types)


def _summary_sections(
    df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw, n_roi
):
    """Return ``[(height, draw_fn), ...]`` for each panel to place in the page.

    ``draw_fn`` takes a target SubFigure and renders one section into it.
    """
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamp"].to_numpy(float)
    rews = df_events.loc[df_events["canonical"] == "Reward", "timestamp"].to_numpy(float)
    row_h = 3.0 * max(n_roi, 1)
    sections = []

    if "session" in want:
        sections.append(
            (
                3.0 + 1.2 * max(n_roi, 1),
                lambda sf: plot_session_overview(
                    df_events, df_fip, paradigm, meta, channels, fig=sf
                ),
            )
        )
    if "psth" in want:
        for cs in paradigm["cs_list"]:
            sections.append(
                (
                    row_h + 1.2,
                    lambda sf, cs=cs: plot_cs_psth_grid(
                        df_fip, cs, cls, meta, channels, fig=sf, **psth_kw
                    ),
                )
            )
    if "lick" in want and len(licks) and len(paradigm["cs_list"]):
        sections.append((4.0, lambda sf: plot_lick_quant(df_events, paradigm, cls, fig=sf)))
    if "antilick" in want and len(licks) and len(paradigm["cs_list"]):
        sections.append(
            (
                4.0,
                lambda sf: plot_anticipatory_lick_summary(
                    df_events, paradigm, cls, meta, fig=sf, **antilick_kw
                ),
            )
        )
    if "rt" in want and len(rews) and len(licks):
        sections.append((4.0, lambda sf: plot_reaction_time(df_events, fig=sf)))
    return sections


def render_summary_figure(
    df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw
):
    """Compose all requested panels into ONE tall figure and return it.

    Returns ``None`` if there is nothing to draw.
    """
    pairs = _channels_present(df_fip, channels)
    n_roi = len({r for _, r in pairs})
    n_chan = len({c for c, _ in pairs})

    sections = _summary_sections(
        df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw, n_roi
    )
    if not sections:
        return None

    heights = [h for h, _ in sections]
    width = max(20.0, 4.5 * max(n_chan, 1))
    fig = plt.figure(figsize=(width, sum(heights)), layout="constrained")
    fig.suptitle(
        "%s  %s   [%s]" % (meta["subject_id"], meta["date"], paradigm["stage"]),
        fontsize=15,
        fontweight="bold",
    )

    subfigs = fig.subfigures(len(sections), 1, height_ratios=heights)
    if len(sections) == 1:
        subfigs = [subfigs]
    for (_, draw_fn), sf in zip(sections, subfigs):
        draw_fn(sf)
    return fig


def _build_summary(paradigm, cls, meta, chan_labels, n_roi):
    """Assemble (and print) the numeric summary dict for a session."""
    summary = {
        "subject_id": meta["subject_id"],
        "date": meta["date"],
        "stage": paradigm["stage"],
        "n_roi": n_roi,
        "channels": chan_labels,
        "adjust_time": meta["adjust_time"],
        "cs": {},
    }
    print(
        "[detected] %s | CS=%s | ROIs=%d | channels=%s | adjust_time=%s"
        % (paradigm["stage"], paradigm["cs_list"], n_roi, chan_labels, meta["adjust_time"])
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
    antilick_window=ANTILICK_WINDOW,
    antilick_summary=ANTILICK_SUMMARY,
    antilick_swarm_width=ANTILICK_SWARM_WIDTH,
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
        If given, the combined single-page summary is written here as a PDF,
        and a PNG is written alongside with the same stem (``.png``).
    plot_types : list or None
        ``['all_sess']``/``['all']`` -> everything, or a subset of
        ``{'session','psth','lick','antilick','rt'}``.
    antilick_window : tuple
        ``(start, end)`` seconds after CS onset for the anticipatory-lick count.
    antilick_summary : str
        ``'mean'`` (mean +/- SD) or ``'median'`` (median +/- IQR).

    Returns
    -------
    dict
        Summary with stage, per-CS trial counts, channels, and roi count.
        When saved, ``pdf`` and ``png`` keys hold the output paths.
    """
    df_events, df_fip, meta = load_pavlovian_dfs(nwb_or_path, preprocessing, adjust_time)
    paradigm = detect_paradigm(df_events)
    cls = classify_trials(df_events, paradigm["cs_list"])

    want = _resolve_want(plot_types)
    pairs = _channels_present(df_fip, channels)
    chan_labels = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    n_roi = len({r for _, r in pairs})

    summary = _build_summary(paradigm, cls, meta, chan_labels, n_roi)

    # anticipatory-lick numbers into the per-CS summary (also lands in the JSON)
    anti = _anticipatory_summary(df_events, paradigm, cls, antilick_window)
    for cs, rec in anti.items():
        summary["cs"].get(cs, {}).update(rec)

    if save_path:
        psth_kw = {
            "t_before": t_before,
            "t_after": t_after,
            "baseline": baseline,
            "output_sampling_rate": output_sampling_rate,
        }
        antilick_kw = {
            "window_s": antilick_window,
            "summary": antilick_summary,
            "swarm_width": antilick_swarm_width,
        }
        fig = render_summary_figure(
            df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw
        )
        if fig is not None:
            pdf_path = save_path if save_path.endswith(".pdf") else save_path + ".pdf"
            png_path = pdf_path[:-4] + ".png"
            fig.savefig(pdf_path)
            fig.savefig(png_path, dpi=PNG_DPI)
            plt.close(fig)
            summary["pdf"] = pdf_path
            summary["png"] = png_path
            print("[saved] %s" % pdf_path)
            print("[saved] %s" % png_path)

    return summary
