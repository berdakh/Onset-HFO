"""The structured, cited report.

Five rules, inherited from the Onset project and enforced by this schema:

1. **Every number carries the window it came from.** A finding without
   evidence windows cannot be built.
2. **Disagreement is reported, not resolved.** Where the two detectors rank a
   channel differently, the report says so; it never averages them into one
   comfortable number.
3. **Data quality is part of the output**, not a footnote: how many channels
   were dropped, how many detections were rejected as artifacts, and why.
4. **Limitations are stated in the report itself**, because that is the page
   somebody will screenshot.
5. **There is no recommendation field.** Not empty -- absent. A schema with a
   "suggested resection" slot will eventually get one filled in.

The report is a plain dataclass: it serialises to JSON for the agent, and
renders to Markdown for a human.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from onset_hfo.config import PIPELINE_VERSION

__all__ = ["EvidenceWindow", "Finding", "Report", "build_report"]


@dataclass
class EvidenceWindow:
    """One inspectable piece of evidence: a time window on a channel.

    ``evidence_id`` is what the agent must cite. It encodes subject, channel,
    detector and start time, so a citation can always be resolved back to the
    signal it came from.
    """

    evidence_id: str
    channel: str
    detector: str
    start: float
    stop: float
    peak_frequency_hz: float
    spectral_prominence_db: float
    amplitude_uv: float
    co_occurs_with_spike: bool = False

    @property
    def duration_ms(self) -> float:
        return (self.stop - self.start) * 1000.0

    def cite(self) -> str:
        return (f"{self.channel} {self.start:.3f}-{self.stop:.3f} s "
                f"[{self.evidence_id}]")


@dataclass
class Finding:
    """One channel, what each detector said about it, and the evidence."""

    channel: str
    contacts: list[str]
    ranks: dict[str, int]
    rates_per_min: dict[str, float]
    rate_ci: dict[str, tuple[float, float]]
    n_events: dict[str, int]
    spike_rate_per_min: float
    fraction_with_spike: float
    evidence: list[EvidenceWindow]
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


@dataclass
class Report:
    """The whole report. No recommendation field, on purpose."""

    recording: dict
    summary: str
    findings: list[Finding]
    disagreements: list[Finding]
    data_quality: list[str]
    methods: list[str]
    limitations: list[str]
    rate_change: list[dict] = field(default_factory=list)
    citation: str = ""
    version: int = 1
    pipeline_version: str = PIPELINE_VERSION
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "recording": self.recording,
            "version": self.version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "findings": [f.as_dict() for f in self.findings],
            "disagreements": [f.as_dict() for f in self.disagreements],
            "rate_change": self.rate_change,
            "data_quality": self.data_quality,
            "methods": self.methods,
            "limitations": self.limitations,
            "citation": self.citation,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=_json_default)

    def evidence_index(self) -> dict[str, EvidenceWindow]:
        """Every citable window in the report, by ``evidence_id``."""
        out: dict[str, EvidenceWindow] = {}
        for finding in list(self.findings) + list(self.disagreements):
            for window in finding.evidence:
                out[window.evidence_id] = window
        return out

    def to_markdown(self) -> str:
        rec = self.recording
        lines = [
            f"# HFO / epileptiform evidence report -- {rec.get('subject', '?')} "
            f"({rec.get('task', '?')}, run {rec.get('run', '?')})",
            "",
            f"*Report v{self.version}, pipeline {self.pipeline_version}, generated {self.generated_at}.*",
            "",
            "> Decision support for research use. This report presents detector output and the",
            "> signal windows behind it. It contains no diagnosis and no treatment recommendation.",
            "",
            "## Summary", "", self.summary, "",
            "## Recording", "",
            f"- source: `{rec.get('source')}`",
            f"- {rec.get('n_channels')} channels analysed, {rec.get('sfreq_hz')} Hz, "
            f"{rec.get('slice_start_s')}-{rec.get('slice_stop_s')} s of the original recording",
        ]
        if rec.get("seizure_onset_s") is not None:
            lines.append(f"- clinician-marked seizure: {rec['seizure_onset_s']}"
                         f"-{rec.get('seizure_offset_s')} s")
        lines += ["", "## Findings", ""]
        if not self.findings:
            lines.append("No channel met the reporting criteria.")
        for f in self.findings:
            lines.append(f"### {f.channel}")
            for det, rate in f.rates_per_min.items():
                lo, hi = f.rate_ci.get(det, (float('nan'), float('nan')))
                lines.append(f"- **{det}**: {rate:.1f} events/min "
                             f"(95% CI {lo:.1f}-{hi:.1f}, n={f.n_events.get(det, 0)}), "
                             f"rank {f.ranks.get(det, '-')}")
            lines.append(f"- interictal discharges: {f.spike_rate_per_min:.1f}/min; "
                         f"{100 * f.fraction_with_spike:.0f}% of its HFOs coincide with one")
            if f.evidence:
                lines.append("- evidence windows: " + "; ".join(e.cite() for e in f.evidence))
            if f.note:
                lines.append(f"- note: {f.note}")
            lines.append("")
        lines += ["## Where the detectors disagree", ""]
        if not self.disagreements:
            lines.append("No channel in the leading group is ranked very differently by the two detectors.")
        for f in self.disagreements:
            ranks = ", ".join(f"{d}: {r}" for d, r in f.ranks.items())
            lines.append(f"- **{f.channel}** -- {ranks}. {f.note}")
            if f.evidence:
                lines.append("  - evidence: " + "; ".join(e.cite() for e in f.evidence[:2]))
        lines.append("")
        if self.rate_change:
            lines += ["## Rate before and during the marked seizure", "",
                      "| channel | before (/min) | during (/min) | ratio |", "|---|---|---|---|"]
            for row in self.rate_change:
                ratio = row.get("ratio")
                ratio_s = "n/a" if ratio is None or (isinstance(ratio, float) and not np.isfinite(ratio)) \
                    else f"{ratio:.1f}x"
                lines.append(f"| {row['channel']} | {row.get('rate_before', 0):.1f} | "
                             f"{row.get('rate_during', 0):.1f} | {ratio_s} |")
            lines.append("")
        lines += ["## Data quality", ""] + [f"- {q}" for q in self.data_quality]
        lines += ["", "## Method", ""] + [f"- {m}" for m in self.methods]
        lines += ["", "## Limitations", ""] + [f"- {lim}" for lim in self.limitations]
        if self.citation:
            lines += ["", "## Data citation", "", self.citation]
        return "\n".join(lines) + "\n"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# --------------------------------------------------------------------------
# Building the report from pipeline tables
# --------------------------------------------------------------------------


def _evidence_id(subject: str, channel: str, detector: str, start: float) -> str:
    return f"{subject}|{channel}|{detector}|{start:.3f}"


def build_report(*, provenance: dict, rates: dict[str, pd.DataFrame], comparison: pd.DataFrame,
                 events: dict[str, list], spike_rates: pd.DataFrame, duration_s: float,
                 preprocessing_steps: list[str], config: dict, rejections: dict[str, dict],
                 rate_change_rows: list[dict] | None = None, top_k: int = 5,
                 evidence_per_channel: int = 3, notes: list[str] | None = None,
                 citation: str = "") -> Report:
    """Assemble a :class:`Report` from the tables the pipeline produced.

    Findings are the channels that at least one detector puts in its top
    ``top_k``; for each, the report carries both detectors' rates and ranks,
    so a channel that one detector likes and the other does not is visible as
    such rather than hidden behind an average.
    """
    subject = provenance.get("subject", "unknown")
    detectors = list(rates)
    ranked = {name: df.set_index("channel") for name, df in rates.items()}
    rank_lookup: dict[str, dict[str, int]] = {}
    for name, df in rates.items():
        order = df.sort_values("rate_per_min", ascending=False).reset_index(drop=True)
        rank_lookup[name] = {ch: int(i + 1) for i, ch in enumerate(order["channel"])}

    spike_lookup = spike_rates.set_index("channel")["rate_per_min"].to_dict() if len(spike_rates) else {}

    candidates: list[str] = []
    for name in detectors:
        for ch, rank in sorted(rank_lookup[name].items(), key=lambda kv: kv[1]):
            if rank <= top_k and ch not in candidates:
                candidates.append(ch)

    disagreeing = set(comparison.loc[comparison["disagrees"], "channel"]) if len(comparison) else set()

    findings: list[Finding] = []
    disagreements: list[Finding] = []
    for ch in candidates:
        finding = _finding_for(ch, detectors, ranked, rank_lookup, events, spike_lookup,
                               subject, evidence_per_channel)
        if ch in disagreeing:
            row = comparison[comparison["channel"] == ch].iloc[0]
            gap = int(row["rank_gap"])
            finding.note = (f"the two detectors' ranks differ by {gap} places; "
                            "the report states both and resolves neither")
            disagreements.append(finding)
        else:
            findings.append(finding)

    leader = findings[0].channel if findings else (disagreements[0].channel if disagreements else None)
    n_ch = provenance.get("n_channels", "?")
    summary_bits = [
        f"{len(detectors)} detectors ran on {provenance.get('subject')} "
        f"({provenance.get('task')}, run {provenance.get('run')}), "
        f"{duration_s:.0f} s of recording, {n_ch} channels.",
    ]
    if leader:
        rate_strings = ", ".join(
            f"{d} {findings[0].rates_per_min.get(d, 0):.1f}/min" if findings else ""
            for d in detectors) if findings else ""
        summary_bits.append(f"Highest event rate: {leader}" + (f" ({rate_strings})." if rate_strings else "."))
    summary_bits.append(f"{len(disagreements)} of the leading channels are ranked very "
                        f"differently by the two detectors.")
    summary_bits.append("This report presents evidence windows and rates; it contains no "
                        "diagnosis and no recommendation.")

    data_quality = list(notes or [])
    for name, counts in rejections.items():
        total = sum(counts.values())
        rejected = total - counts.get("accepted", 0)
        data_quality.append(
            f"{name}: {total} candidate events, {rejected} rejected by artifact validation "
            f"({', '.join(f'{k}: {v}' for k, v in counts.items() if k != 'accepted') or 'none'})")
    data_quality.append("rates are events per minute over the analysed window; "
                        "confidence intervals assume a Poisson process")

    methods = list(preprocessing_steps) + [
        f"detectors: {', '.join(detectors)} (see docs/METHODS.md)",
        f"configuration: {json.dumps(config, default=str)}",
    ]

    has_markings = bool(provenance.get("n_expert_events"))
    limitations = [
        "prototype: thresholds come from the literature, checked on synthetic data and "
        "against expert markings on 20 subjects of ds003498; not validated on a clinical cohort",
        ("this recording carries expert HFO markings, so the detectors can be scored on it "
         "(python -m onset_hfo.cli benchmark); agreement with those markings is not accuracy, "
         "because the reference is another detector's validated output"
         if has_markings else
         "this recording carries no HFO markings, so precision and recall cannot be computed "
         "on it; they are measured on ds003498 and on synthetic data -- see docs/EVALUATION.md"),
        "event rate is not a diagnosis: physiological ripples occur in healthy tissue, "
        "particularly in mesial temporal and occipital regions",
        "a single short window of a single patient; nothing here generalises",
        "bipolar pairs are formed from consecutive contact numbers, which on a grid is not "
        "always spatial adjacency",
    ]
    if provenance.get("sfreq_hz", 0) < 600:
        limitations.append(f"sampling rate {provenance['sfreq_hz']:g} Hz allows ripples (80-250 Hz) "
                           "only; fast ripples were not analysed")

    return Report(
        recording=provenance,
        summary=" ".join(b for b in summary_bits if b),
        findings=findings,
        disagreements=disagreements,
        data_quality=data_quality,
        methods=methods,
        limitations=limitations,
        rate_change=list(rate_change_rows or []),
        citation=citation,
    )


def _finding_for(ch: str, detectors: list[str], ranked: dict[str, pd.DataFrame],
                 rank_lookup: dict[str, dict[str, int]], events: dict[str, list],
                 spike_lookup: dict[str, float], subject: str, k: int) -> Finding:
    rates, cis, counts = {}, {}, {}
    contacts: list[str] = []
    for name in detectors:
        table = ranked[name]
        if ch in table.index:
            row = table.loc[ch]
            rates[name] = float(row["rate_per_min"])
            cis[name] = (float(row.get("rate_ci_low", float("nan"))),
                         float(row.get("rate_ci_high", float("nan"))))
            counts[name] = int(row["n_events"])
        else:
            rates[name], cis[name], counts[name] = 0.0, (0.0, 0.0), 0

    evidence: list[EvidenceWindow] = []
    spike_overlap = []
    for name in detectors:
        best = sorted([e for e in events.get(name, []) if e.channel == ch and e.accepted],
                      key=lambda e: -e.score)[:k]
        for e in best:
            evidence.append(EvidenceWindow(
                evidence_id=_evidence_id(subject, ch, name, e.start),
                channel=ch, detector=name, start=round(e.start, 3), stop=round(e.stop, 3),
                peak_frequency_hz=round(float(e.peak_frequency_hz), 1)
                if np.isfinite(e.peak_frequency_hz) else float("nan"),
                spectral_prominence_db=round(float(e.spectral_prominence_db), 1)
                if np.isfinite(e.spectral_prominence_db) else float("nan"),
                amplitude_uv=round(float(e.peak_amplitude_uv), 1),
                co_occurs_with_spike=bool(e.co_occurs_with_spike)))
            if not contacts:
                contacts = list(e.contacts)
        spike_overlap += [e.co_occurs_with_spike for e in events.get(name, [])
                          if e.channel == ch and e.accepted]

    fraction = float(np.mean(spike_overlap)) if spike_overlap else 0.0
    return Finding(
        channel=ch, contacts=contacts or [ch],
        ranks={n: rank_lookup[n].get(ch, -1) for n in detectors},
        rates_per_min=rates, rate_ci=cis, n_events=counts,
        spike_rate_per_min=float(spike_lookup.get(ch, 0.0)),
        fraction_with_spike=fraction,
        evidence=sorted(evidence, key=lambda e: e.start),
    )
