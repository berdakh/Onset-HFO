"""Figures.

Four figures answer the four questions someone actually asks of this
pipeline:

1. *What does one detection look like?* -> :func:`plot_event`
2. *Which channels stand out?* -> :func:`plot_channel_rates`
3. *Does the rate change over time (e.g. at seizure onset)?* -> :func:`plot_rate_timecourse`
4. *Do the two detectors agree?* -> :func:`plot_detector_comparison`

Style rules followed here: one measure per axis (never two y-scales), a
legend whenever more than one series is drawn, direct labels instead of a
number on every mark, recessive grid and axes, and a fixed categorical colour
order rather than a cycled palette. Colours are the validated default
categorical slots (blue / orange / aqua) plus a reserved status red for
rejected events.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.detectors.base import Event, bandpass
from onset_hfo.preprocess import Prepared

__all__ = ["PALETTE", "plot_event", "plot_channel_rates", "plot_rate_timecourse",
           "plot_detector_comparison", "save_all_figures"]

#: Fixed categorical order (never cycled) plus reserved status colours.
PALETTE = {
    "series_1": "#2a78d6",   # blue   -- first detector / primary measure
    "series_2": "#eb6834",   # orange -- second detector
    "series_3": "#1baf7a",   # aqua   -- third series, if ever needed
    "rejected": "#e34948",   # status red, reserved for rejected events
    "seizure": "#f0efec",    # neutral shading for the marked seizure
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
    "grid": "#dcdbd6",
    "surface": "#fcfcfb",
}

DETECTOR_COLORS = {"rms": PALETTE["series_1"], "line_length": PALETTE["series_2"],
                   "spike": PALETTE["series_3"]}


def _fig(*args, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(PALETTE["surface"])
    for axis in np.atleast_1d(ax).ravel():
        _style(axis)
    return fig, ax


def _style(ax) -> None:
    ax.set_facecolor(PALETTE["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["ink_soft"], labelsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def plot_event(prep: Prepared, event: Event, context_s: float = 0.4, band: tuple | None = None):
    """Three stacked views of one event: wideband, band-passed, and spectrum.

    This is the figure that makes a detection arguable: a reader can see
    whether the "ripple" is an oscillation with a spectral bump of its own, or
    a sharp transient that the filter turned into one.
    """
    import matplotlib.pyplot as plt

    band = tuple(band or event.band)
    sf = prep.sfreq
    i_mid = int(round((event.mid - prep.t_offset) * sf))
    half = int(round(context_s * sf / 2))
    i0, i1 = max(0, i_mid - half), min(prep.data.shape[1], i_mid + half)
    row = prep.ch_names.index(event.channel)
    raw = prep.data[row, i0:i1]
    filt = bandpass(prep.data[row:row + 1, i0:i1], sf, band)[0]
    t = prep.t_offset + np.arange(i0, i1) / sf

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5),
                             gridspec_kw={"height_ratios": [2, 2, 2.2]})
    fig.patch.set_facecolor(PALETTE["surface"])
    for ax in axes:
        _style(ax)
    colour = DETECTOR_COLORS.get(event.detector, PALETTE["series_1"])
    mark = colour if event.accepted else PALETTE["rejected"]

    axes[0].plot(t, raw, linewidth=1.0, color=PALETTE["ink"])
    axes[0].set_ylabel("wideband (uV)", color=PALETTE["ink_soft"], fontsize=9)
    axes[1].plot(t, filt, linewidth=1.2, color=colour)
    axes[1].set_ylabel(f"{band[0]:g}-{band[1]:g} Hz (uV)", color=PALETTE["ink_soft"], fontsize=9)
    for ax in axes[:2]:
        ax.axvspan(event.start, event.stop, color=mark, alpha=0.16, linewidth=0)
        ax.set_xlim(t[0], t[-1])

    # Spectrum of the event window against the surrounding background: the
    # "is there really a bump" question, drawn.
    from onset_hfo.spectral import background_level_db, event_psd

    j0 = max(0, int(round((event.start - prep.t_offset) * sf)) - i0)
    j1 = min(len(raw), int(round((event.stop - prep.t_offset) * sf)) - i0 + 1)
    freqs, psd = event_psd(raw[max(0, j0 - int(0.05 * sf)):j1 + int(0.05 * sf)], sf)
    if freqs.size:
        axes[2].plot(freqs, 10 * np.log10(np.maximum(psd, 1e-30)), linewidth=1.5,
                     color=colour, label="event window")
        background = background_level_db(freqs, psd, band)
        if background is not None:
            axes[2].plot(freqs, background, linewidth=1.2, linestyle="--",
                         color=PALETTE["ink_soft"], label="fitted 1/f background")
        axes[2].axvspan(band[0], band[1], color=PALETTE["series_1"], alpha=0.07, linewidth=0)
        axes[2].set_xlim(0, min(sf / 2, band[1] * 2))
        axes[2].legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink_soft"])
    axes[2].set_xlabel("frequency (Hz)", color=PALETTE["ink_soft"], fontsize=9)
    axes[2].set_ylabel("power (dB)", color=PALETTE["ink_soft"], fontsize=9)
    axes[1].set_xlabel("time (s, original recording)", color=PALETTE["ink_soft"], fontsize=9)

    status = "accepted" if event.accepted else f"rejected: {event.reject_reason}"
    title = (f"{event.channel} - {event.detector} - {event.start:.3f}-{event.stop:.3f} s "
             f"({event.duration_ms:.0f} ms)")
    subtitle = (f"peak {event.peak_frequency_hz:.0f} Hz, "
                f"{event.spectral_prominence_db:.1f} dB over background, "
                f"{event.peak_amplitude_uv:.0f} uV, {event.n_peaks} rectified peaks - {status}")
    fig.suptitle(title, fontsize=12, color=PALETTE["ink"], x=0.01, ha="left")
    axes[0].set_title(subtitle, fontsize=9, color=PALETTE["ink_soft"], loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_channel_rates(rates: pd.DataFrame, top_k: int = 15, label: str = "rms",
                       title: str | None = None):
    """Ranked per-channel event rate, with Poisson confidence intervals.

    Horizontal bars, because channel names are text; intervals drawn, because
    a rate from a one-minute window is a small count wearing a confident face.
    """
    data = rates.head(top_k).iloc[::-1]
    fig, ax = _fig(figsize=(8, max(3.0, 0.32 * len(data) + 1.4)))
    y = np.arange(len(data))
    colour = DETECTOR_COLORS.get(label, PALETTE["series_1"])
    ax.barh(y, data["rate_per_min"], height=0.62, color=colour, edgecolor=PALETTE["surface"],
            linewidth=2)
    if {"rate_ci_low", "rate_ci_high"} <= set(data.columns):
        ax.errorbar(data["rate_per_min"], y,
                    xerr=[data["rate_per_min"] - data["rate_ci_low"],
                          data["rate_ci_high"] - data["rate_per_min"]],
                    fmt="none", ecolor=PALETTE["ink_soft"], elinewidth=1, capsize=3, alpha=0.7)
    ax.set_yticks(y, data["channel"], fontsize=9)
    ax.set_xlabel(f"{label} events per minute", color=PALETTE["ink_soft"], fontsize=9)
    ax.grid(axis="y", visible=False)
    for yi, (rate, n) in enumerate(zip(data["rate_per_min"], data["n_events"], strict=False)):
        ax.text(rate, yi, f"  {rate:.1f}  (n={int(n)})", va="center", fontsize=8,
                color=PALETTE["ink_soft"])
    ax.set_xlim(0, float(data["rate_ci_high"].max() if "rate_ci_high" in data else
                         data["rate_per_min"].max()) * 1.25 + 0.5)
    ax.set_title(title or f"Event rate by channel ({label})", fontsize=12,
                 color=PALETTE["ink"], loc="left")
    fig.tight_layout()
    return fig


def plot_rate_timecourse(events: dict[str, list[Event]], t_start: float, t_stop: float,
                         bin_s: float = 5.0, channels: list[str] | None = None,
                         seizure: tuple[float | None, float | None] = (None, None),
                         title: str | None = None):
    """Event rate over time, one line per detector, seizure window shaded."""
    from onset_hfo.metrics import rate_timecourse

    fig, ax = _fig(figsize=(9, 4))
    on, off = seizure
    if on is not None:
        ax.axvspan(on, off if off is not None else t_stop, color=PALETTE["seizure"],
                   linewidth=0, zorder=0)
        ax.text(on, ax.get_ylim()[1], " marked seizure", fontsize=8, va="top",
                color=PALETTE["ink_soft"])
    for name, evs in events.items():
        tc = rate_timecourse(evs, t_start, t_stop, bin_s, channels)
        ax.plot(tc["t_start"] + bin_s / 2, tc["rate_per_min"], linewidth=2,
                color=DETECTOR_COLORS.get(name, PALETTE["series_3"]), label=name)
    ax.set_xlabel("time (s, original recording)", color=PALETTE["ink_soft"], fontsize=9)
    ax.set_ylabel("events per minute", color=PALETTE["ink_soft"], fontsize=9)
    if len(events) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=PALETTE["ink_soft"])
    scope = "all channels" if not channels else ", ".join(channels[:4]) + \
        ("..." if len(channels) > 4 else "")
    ax.set_title(title or f"Event rate over time ({scope})", fontsize=12,
                 color=PALETTE["ink"], loc="left")
    fig.tight_layout()
    return fig


def plot_detector_comparison(comparison: pd.DataFrame, name_a: str = "rms",
                             name_b: str = "line_length", label_top: int = 6):
    """Rate from one detector against the other, one point per channel.

    Points on the diagonal are channels the two detectors see the same way.
    Points the report calls disagreements are drawn in the status colour and
    labelled: this figure exists so that a disagreement is impossible to miss.
    """
    fig, ax = _fig(figsize=(6.2, 6))
    x = comparison[f"rate_per_min_{name_a}"]
    y = comparison[f"rate_per_min_{name_b}"]
    agree = ~comparison["disagrees"]
    ax.plot([0, max(x.max(), y.max()) * 1.05], [0, max(x.max(), y.max()) * 1.05],
            linewidth=1, linestyle="--", color=PALETTE["grid"], zorder=0)
    ax.scatter(x[agree], y[agree], s=46, color=PALETTE["series_1"],
               edgecolor=PALETTE["surface"], linewidth=1.5, label="same ranking")
    if (~agree).any():
        ax.scatter(x[~agree], y[~agree], s=64, color=PALETTE["rejected"],
                   edgecolor=PALETTE["surface"], linewidth=1.5, marker="D",
                   label="ranked very differently")
    for _, row in comparison.head(label_top).iterrows():
        ax.annotate(row["channel"], (row[f"rate_per_min_{name_a}"], row[f"rate_per_min_{name_b}"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=8,
                    color=PALETTE["ink_soft"])
    ax.set_xlabel(f"{name_a} events per minute", color=PALETTE["ink_soft"], fontsize=9)
    ax.set_ylabel(f"{name_b} events per minute", color=PALETTE["ink_soft"], fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=PALETTE["ink_soft"])
    ax.set_title("Do the two detectors agree?", fontsize=12, color=PALETTE["ink"], loc="left")
    fig.tight_layout()
    return fig


def save_all_figures(result, directory: str | Path, top_k: int = 15, dpi: int = 140) -> list[Path]:
    """Render the standard figure set for a pipeline result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    prep = result.prepared
    t0, t1 = prep.t_offset, prep.t_offset + prep.duration

    for name, table in result.rates.items():
        fig = plot_channel_rates(table, top_k=top_k, label=name)
        path = out / f"rates_{name}.png"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(path)

    top = list(result.rates[list(result.rates)[0]]["channel"].head(3))
    fig = plot_rate_timecourse(result.events, t0, t1, channels=top,
                               seizure=result.recording.seizure)
    path = out / "rate_timecourse.png"
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    written.append(path)

    if len(result.rates) >= 2 and len(result.comparison):
        names = list(result.rates)
        fig = plot_detector_comparison(result.comparison, names[0], names[1])
        path = out / "detector_comparison.png"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(path)

    best = result.evidence(top[0], k=1) if top else []
    if best:
        fig = plot_event(prep, best[0])
        path = out / "example_event.png"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(path)
    rejected = [e for evs in result.events.values() for e in evs if not e.accepted]
    if rejected:
        fig = plot_event(prep, sorted(rejected, key=lambda e: -e.score)[0])
        path = out / "example_rejected_event.png"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(path)
    return written
