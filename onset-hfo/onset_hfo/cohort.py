"""Clinician SOZ labels, surgical outcome and recording site, from the archive.

Why this module exists
----------------------
Scoring a channel ranking needs a reference standard. Until now this
repository had only the *free-text contact names* a reviewer typed into the
events file during a seizure -- a weak textual reference that
``docs/DATA.md`` is careful not to oversell.

It turns out ``ds003029`` also publishes the curated version, in
``sourcedata/clinical_data_summary.xlsx``: one row per patient with the
clinician-defined **seizure onset zone contacts**, the **Engel** and **ILAE**
scores, whether a resection or ablation was performed, and the **clinical
centre**. It is CC0 like the rest of the archive and it is 30 KB.

That single file turns three things from "roadmap" into "runnable today":

* a contact-level reference standard for any ranking this pipeline produces;
* surgical outcome, so a ranking can be scored where the label is trustworthy
  (a resected contact in a patient who became seizure-free);
* four recording sites (NIH, JHH, UMMC, UMF), which is what a leave-one-site-
  out generalization study needs.

Two traps, both of which silently corrupt a study
-------------------------------------------------
**The ``outcome`` column is not what it looks like.** ``S`` means *success*
(seizure-free; every ``S`` row carries Engel 1) and ``F`` means *failure*
(Engel 2-4). Reading ``F`` as "free" inverts every label in the cohort.
:func:`decode_outcome` is the only place that mapping is written down, and
``test_cohort.py`` pins it against the Engel scores.

**Subject ids do not match between the archive and the spreadsheet.** The S3
tree has ``sub-pt01`` while the spreadsheet says ``pt1``; it also has a
separate, nearly empty ``sub-pt1``. :func:`normalize_subject` reconciles them
by stripping the ``sub-`` prefix and the zero padding.

Swapping in your own labels
---------------------------
Everything downstream depends only on :class:`SozLabels`, so a local cohort
substitutes for the public one without touching the pipeline or the agent::

    from onset_hfo.cohort import SozLabels, load_labels_csv

    labels = load_labels_csv("our-centre/soz.csv")      # subject,soz_contacts,engel,...
    labels = SozLabels(subject="anon-01", soz_contacts={"LA1", "LA2"}, engel=1)

:func:`soz_labels` prefers, in order: an explicit CSV you pass, the archive
spreadsheet, and finally the free-text markers already in the events file.
The last of those is clearly marked ``source="events_markers"`` so that a
result computed against it is never mistaken for one computed against a
curated label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from onset_hfo.config import DATA_CACHE, DATASET

__all__ = [
    "SozLabels",
    "expand_contacts",
    "normalize_subject",
    "decode_outcome",
    "clinical_table",
    "soz_labels",
    "load_labels_csv",
    "label_channels",
    "cohort_table",
]

#: Filename inside the archive. CC0, ~30 KB, one row per patient.
CLINICAL_XLSX = "sourcedata/clinical_data_summary.xlsx"

#: ``outcome`` column -> "did this patient become seizure free?".
#: S = success, F = failure, NR = no resection performed. See the module
#: docstring: the naive reading of ``F`` is wrong and inverts the cohort.
_OUTCOME = {"S": True, "F": False, "NR": None}

_RANGE_RE = re.compile(r"^([A-Za-z][A-Za-z']*)\s*(\d+)\s*[-–—]\s*(\d+)$")
_SINGLE_RE = re.compile(r"^([A-Za-z][A-Za-z']*)\s*(\d+)$")
_PAD_RE = re.compile(r"^([a-z]+?)0*(\d+)$")
#: A contact range wider than this is a parsing accident, not an electrode.
MAX_RANGE = 200


# --------------------------------------------------------------------------
# Parsing the clinician's shorthand
# --------------------------------------------------------------------------


def expand_contacts(text: str | float | None) -> set[str]:
    """Expand clinician shorthand into contact names, upper-cased.

    The spreadsheet's ``soz_contacts`` field is written for a human:
    ``"TT1-6; AST1-2, mst1-2"``, ``"RPG4-5, RPG12-14; APD1-8"``, sometimes with
    a trailing newline. Separators are ``;``, ``,``, ``:`` and newline -- the
    colon appears where a reviewer wrote ``"PST1-4: AST1-2"``, so treating it
    as punctuation rather than a separator silently loses both groups. A token
    is either a single contact (``G16``) or an inclusive range on one electrode
    (``TT1-6``).

    Case is discarded because the same electrode appears as ``mst1`` in one
    row and ``MST1`` in the next, while ``channels.tsv`` uses its own casing.

    >>> sorted(expand_contacts("TT1-3; AST1, mst2"))
    ['AST1', 'MST2', 'TT1', 'TT2', 'TT3']
    """
    out: set[str] = set()
    if text is None or isinstance(text, float) or not str(text).strip():
        return out
    for token in re.split(r"[;,:\n]", str(text)):
        token = token.strip()
        if not token:
            continue
        match = _RANGE_RE.match(token)
        if match:
            prefix, first, last = match.group(1), int(match.group(2)), int(match.group(3))
            if 0 <= last - first < MAX_RANGE:
                out |= {f"{prefix}{i}".upper() for i in range(first, last + 1)}
            continue
        match = _SINGLE_RE.match(token)
        if match:
            out.add(f"{match.group(1)}{match.group(2)}".upper())
    return out


def normalize_subject(subject: str) -> str:
    """``"sub-pt01"`` and ``"pt1"`` both become ``"pt1"``.

    The archive's directory names are zero-padded for some centres and not for
    others; the spreadsheet is never padded. Normalising both sides is the
    whole of the reconciliation.
    """
    key = str(subject).strip().lower()
    key = key[4:] if key.startswith("sub-") else key
    return _PAD_RE.sub(lambda m: f"{m.group(1)}{int(m.group(2))}", key)


def decode_outcome(code: str | float | None) -> bool | None:
    """``"S"`` -> True (seizure free), ``"F"`` -> False, ``"NR"``/blank -> None.

    Returning ``None`` rather than ``False`` for "no resection" matters: those
    patients have no surgical ground truth at all, and treating them as
    failures would put unlabelled contacts into the negative class.
    """
    if code is None or isinstance(code, float) or not str(code).strip():
        return None
    return _OUTCOME.get(str(code).strip().upper())


# --------------------------------------------------------------------------
# The labels themselves
# --------------------------------------------------------------------------


@dataclass
class SozLabels:
    """The reference standard for one patient.

    Attributes
    ----------
    soz_contacts:
        Contacts the clinician identified as seizure onset, upper-cased.
    engel / ilae:
        Surgical outcome scales. ``None`` when not recorded (the archive uses
        ``-1`` for this, which is decoded away here so that nobody averages it).
    seizure_free:
        ``True`` (Engel I), ``False``, or ``None`` when no resection happened.
    site:
        Recording centre -- the grouping variable for a leave-one-site-out
        generalization study.
    source:
        Where the labels came from: ``"clinical_summary"`` (curated),
        ``"events_markers"`` (weak free text), ``"local"`` (your own CSV), or
        ``"none"``. Carry this into every result table; a number computed
        against weak labels must never be reported as if it were curated.
    """

    subject: str
    soz_contacts: set[str] = field(default_factory=set)
    engel: int | None = None
    ilae: float | None = None
    seizure_free: bool | None = None
    site: str = ""
    modality: str = ""
    surgery_type: str = ""
    source: str = "none"

    @property
    def usable(self) -> bool:
        """True when there is at least one labelled contact to score against."""
        return bool(self.soz_contacts)

    @property
    def trustworthy(self) -> bool:
        """True when the label is curated *and* the surgery worked.

        This is the subset both thesis proposals name as the honest positive
        class: a contact the clinician called onset, in a patient who actually
        became seizure free. Everywhere else the label is a hypothesis.
        """
        return self.usable and self.source == "clinical_summary" and self.seizure_free is True

    def as_dict(self) -> dict:
        return {"subject": self.subject, "n_soz_contacts": len(self.soz_contacts),
                "soz_contacts": sorted(self.soz_contacts), "engel": self.engel,
                "ilae": self.ilae, "seizure_free": self.seizure_free, "site": self.site,
                "modality": self.modality, "surgery_type": self.surgery_type,
                "source": self.source, "trustworthy": self.trustworthy}


# --------------------------------------------------------------------------
# Reading the archive
# --------------------------------------------------------------------------


def _clinical_url() -> str:
    return f"{DATASET.base_url}/{DATASET.dataset_id}/{CLINICAL_XLSX}"


def clinical_table(cache_dir: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download (once) and return the archive's clinical summary sheet.

    Cached beside the signal slices, so a second call is instant and an
    offline session keeps working.
    """
    cache = Path(cache_dir or DATA_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "clinical_data_summary.xlsx"
    if refresh or not local.exists():
        from onset_hfo.datasets import _http_get  # local import: keeps this module importable
        local.write_bytes(_http_get(_clinical_url()))
    frame = pd.read_excel(local)
    frame["_key"] = frame["dataset_id"].astype(str).map(normalize_subject)
    return frame


def soz_labels(subject: str, csv: str | Path | None = None,
               recording=None, cache_dir: str | Path | None = None) -> SozLabels:
    """Best available labels for one subject, with the source recorded.

    Order of preference, most to least trustworthy:

    1. ``csv`` -- your own curated file (see :func:`load_labels_csv`);
    2. the archive's clinical summary;
    3. ``recording.marked_contacts`` -- the free-text markers, if a
       :class:`~onset_hfo.datasets.Recording` is supplied.

    Never raises for a missing label: it returns ``source="none"`` and an
    empty contact set, so a cohort run can report coverage honestly instead of
    crashing on the first patient without a row.
    """
    if csv is not None:
        table = load_labels_csv(csv)
        if subject in table:
            return table[subject]
        key = normalize_subject(subject)
        for name, labels in table.items():
            if normalize_subject(name) == key:
                return labels

    try:
        frame = clinical_table(cache_dir)
    except Exception:  # offline, or the archive moved: fall through to markers
        frame = None
    if frame is not None:
        rows = frame[frame["_key"] == normalize_subject(subject)]
        if len(rows):
            row = rows.iloc[0]
            engel = _int_or_none(row.get("engel_score"))
            ilae = _float_or_none(row.get("ilae_score"))
            return SozLabels(
                subject=subject,
                soz_contacts=expand_contacts(row.get("soz_contacts")),
                engel=engel if engel is not None and engel > 0 else None,
                ilae=ilae if ilae is not None and ilae > 0 else None,
                seizure_free=decode_outcome(row.get("outcome")),
                site=str(row.get("clinical_center") or "").upper(),
                modality=str(row.get("modality") or ""),
                surgery_type=str(row.get("surgery_type") or ""),
                source="clinical_summary")

    marked = list(getattr(recording, "marked_contacts", []) or [])
    if marked:
        return SozLabels(subject=subject,
                         soz_contacts={str(c).upper() for c in marked},
                         source="events_markers")
    return SozLabels(subject=subject, source="none")


def load_labels_csv(path: str | Path) -> dict[str, SozLabels]:
    """Read a local label file into :class:`SozLabels`, keyed by subject.

    Expected columns: ``subject`` and ``soz_contacts`` (clinician shorthand is
    fine -- it goes through :func:`expand_contacts`). Optional: ``engel``,
    ``ilae``, ``seizure_free``, ``site``, ``modality``, ``surgery_type``.

    This is the seam for a local cohort: produce this one file and every
    metric in the repository works on your patients.
    """
    frame = pd.read_csv(path)
    missing = {"subject", "soz_contacts"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}")
    out: dict[str, SozLabels] = {}
    for _, row in frame.iterrows():
        free = row.get("seizure_free")
        out[str(row["subject"])] = SozLabels(
            subject=str(row["subject"]),
            soz_contacts=expand_contacts(row["soz_contacts"]),
            engel=_int_or_none(row.get("engel")),
            ilae=_float_or_none(row.get("ilae")),
            seizure_free=None if pd.isna(free) else bool(free),
            site=str(row.get("site") or ""),
            modality=str(row.get("modality") or ""),
            surgery_type=str(row.get("surgery_type") or ""),
            source="local")
    return out


# --------------------------------------------------------------------------
# Applying labels to analysed channels
# --------------------------------------------------------------------------


def label_channels(ch_names: list[str], labels: SozLabels,
                   contacts_of=None) -> pd.DataFrame:
    """Mark which analysed channels touch a labelled SOZ contact.

    A bipolar channel is positive when **either** of its contacts is in the
    SOZ set. That is the same rule ``evaluate.py`` already uses to match a
    detection to a truth event, and it is the permissive choice: a pair
    straddling the SOZ border counts as inside it. The alternative (both
    contacts) would be defensible too -- it is stated here rather than hidden
    because it moves every precision number.

    Parameters
    ----------
    contacts_of:
        ``f(channel) -> [contact, ...]``. Pass ``prepared.contacts_of`` when
        you have it; otherwise the channel name is split on ``-``.
    """
    split = contacts_of or (lambda ch: [p for p in str(ch).split("-") if p])
    rows = []
    for ch in ch_names:
        contacts = [str(c).upper() for c in split(ch)]
        hit = sorted(set(contacts) & labels.soz_contacts)
        rows.append({"channel": ch, "contacts": contacts, "is_soz": bool(hit),
                     "matched_contacts": hit})
    return pd.DataFrame(rows)


def cohort_table(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """One row per patient in the clinical summary, decoded and ready to group.

    Use it to choose a cohort before downloading anything: which patients have
    SOZ contacts at all, which became seizure free, and how the sites divide.
    """
    frame = clinical_table(cache_dir)
    rows = []
    for _, row in frame.iterrows():
        contacts = expand_contacts(row.get("soz_contacts"))
        engel = _int_or_none(row.get("engel_score"))
        rows.append({
            "subject": str(row["dataset_id"]),
            "key": row["_key"],
            "site": str(row.get("clinical_center") or "").upper(),
            "modality": str(row.get("modality") or ""),
            "n_soz_contacts": len(contacts),
            "engel": engel if engel is not None and engel > 0 else None,
            "seizure_free": decode_outcome(row.get("outcome")),
            "surgery_type": str(row.get("surgery_type") or ""),
        })
    return pd.DataFrame(rows)


def _int_or_none(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
