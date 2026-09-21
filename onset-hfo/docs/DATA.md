# The data

## What we use, and why

[**OpenNeuro ds003029** — *Epilepsy-iEEG-Multicenter-Dataset*](https://openneuro.org/datasets/ds003029)

| | |
|---|---|
| Licence | **CC0** (public domain dedication) — no data use agreement needed |
| Contents | intracranial EEG (ECoG and SEEG) from ~100 subjects, four clinical centres (JHH, NIH, UMMC, UMF) |
| Format | BIDS-iEEG; signals as BrainVision (`.vhdr` + `.vmrk` + `.eeg`) |
| Sampling | 1000 Hz for the recordings used here |
| Annotations | clinician markers for electrographic seizure onset/offset, and free-text notes naming contacts; per-channel `good`/`bad` status |
| Published with | *Neural Fragility as an EEG Marker of the Seizure Onset Zone* (Li et al., Nature Neuroscience, 2023) |
| DOI | `10.18112/openneuro.ds003029.v1.0.3` |

It was chosen for four reasons: it is genuinely public (CC0, no gatekeeping),
it is intracranial (scalp EEG cannot show ripples), it carries clinician
annotations we can compare against, and its files can be read **in byte
ranges over plain HTTPS**, which is what makes a ten-minute notebook possible.

The default example is `sub-pt01`, `task-ictal`, `run-01`: 98 channels,
1000 Hz, 269 s, with an electrographic seizure marked at 75.95–161.82 s.

## How only 24 MB gets downloaded

BrainVision stores samples **multiplexed**: channel 1 at time 1, channel 2 at
time 1, …, channel N at time 1, then time 2. There is no per-sample header. So
sample *k* begins at byte `k × n_channels × bytes_per_sample`, and a byte range
is a time range.

`onset_hfo.datasets.fetch_slice` therefore:

1. downloads the three small text files (`.vhdr` header, `channels.tsv`,
   `events.tsv`) in full;
2. parses the header for channel count, sampling interval and binary format;
3. issues one HTTP `Range` request for the seconds you asked for;
4. writes that slice into a cache directory next to a copy of the header, plus
   a minimal marker file;
5. reads it with MNE, attaches the annotations (shifted into slice time),
   marks the dataset's `bad` channels, and records `t_offset` so that every
   time the pipeline reports is a time in the **original** recording.

60 seconds of 98 channels at 1000 Hz in 32-bit floats is 23.5 MB. The whole
run is 105 MB; the whole dataset is hundreds of gigabytes.

```python
from onset_hfo.datasets import fetch_slice, list_runs

list_runs("sub-pt01")                       # what exists, with file sizes
rec = fetch_slice("sub-pt01", "ictal", "01", t_start=50, t_stop=110)
rec.provenance()                            # everything needed to cite it
```

Slices are cached under `artifacts/data/`; a second call is instant. Set
`ONSET_HFO_HOME` to move the cache (Colab: leave it alone).

## What the annotations mean — and do not

**Seizure onset and offset** come from the clinician's markers in the events
file. The dataset's own README warns that marker wording is not standardised
across centres, which is why `datasets.py` matches them with regular
expressions rather than exact strings.

**Contact names in the markers.** During a seizure the reviewer types notes:
`"AD1-4, ATT1,2"`, `"G16"`, `"AST1,3"`. `parse_marked_contacts` expands those
into contact names. This is a **weak textual reference**, and the distinction
matters:

* it is *not* a curated seizure-onset zone — that information lives in a
  clinical spreadsheet in the publication, not in this archive;
* it is incomplete, unordered, and written for a human reader;
* a contact may be named for reasons this pipeline does not model.

`onset_hfo.evaluate.marked_contact_check` compares the top-ranked channels
against these contacts with a permutation test. Read the result as mild
encouragement at best, and never as accuracy. See `EVALUATION.md`.

**`status = bad`** in `channels.tsv` marks contacts the dataset authors
excluded: white matter, ventricle, CSF, outside the brain, or noisy. The
pipeline drops them by default, and the report states how many and which.

## One thing this dataset cannot give us

It publishes **ictal** snapshots — recordings around seizures. Clinical HFO
research is usually done on **interictal** data (between seizures, often in
slow-wave sleep), where a high ripple rate is the marker of interest.

So what the prototype measures on this data is "where, in this seizure, the
ripple band is loudest", not the classical interictal HFO rate. The pipeline
partly compensates by comparing the pre-onset portion of the window with the
seizure itself (`rate_change`), but the pre-onset stretch is minutes from a
seizure and is not equivalent to a quiet interictal recording. This is stated
in every report's limitations and is the first item in the roadmap.

## A note on amplitude units

The BrainVision headers declare a resolution of 1 nV per stored unit, and the
loader converts to microvolts on that basis. The resulting background level in
this recording is about 120 µV RMS on a bipolar channel before the seizure,
which is plausible for ECoG but on the high side of the usual 20–100 µV, so
treat absolute microvolt values in the report as "as declared by the archive's
header" rather than as calibrated measurements.

Detection itself is unaffected by a constant scale factor: every threshold in
this pipeline is expressed in robust standard deviations **of the channel
itself**, so multiplying a recording by any constant changes no decision. Only
the absolute µV numbers printed alongside events would move.

## Sampling rate and the fast-ripple question

At 1000 Hz the Nyquist frequency is 500 Hz. Ripples (80–250 Hz) are fine.
**Fast ripples (250–500 Hz) are not analysable**: they sit against the Nyquist
edge, where anti-alias filtering and noise make any measurement untrustworthy.
`config.Bands.usable()` checks this, the pipeline skips the band, and the
report says so. Studies that want fast ripples record at 2000 Hz or more — the
simulator defaults to 2000 Hz for exactly that reason.

## Using your own data

Three ways, in increasing order of effort.

**1. A different recording from the same archive.**

```python
from onset_hfo.datasets import list_runs, fetch_slice
list_runs("sub-jh103")
rec = fetch_slice("sub-jh103", "ictal", "02", t_start=0, t_stop=60)
```

**2. A local BrainVision file.**

```python
from onset_hfo.datasets import load_local_brainvision
rec = load_local_brainvision("/path/to/recording.vhdr", subject="anon-01",
                             seizure=(120.0, 190.0))
```

**3. Any other format MNE can read.** Build a `Recording` yourself — it is a
plain dataclass:

```python
import mne
from onset_hfo.datasets import Recording

raw = mne.io.read_raw_edf("/path/to/recording.edf", preload=True)
rec = Recording(raw=raw, source="local:our-centre", subject="anon-01",
                task="interictal", run="01", t_offset=0.0,
                seizure=(None, None), bads=["A1", "A2"],
                citation="internal, de-identified")
```

The pipeline needs only: a preloaded `Raw` in volts, channel names that follow
the `LETTERS + NUMBER` convention (so neighbouring contacts can be paired),
and — if you want the mains notch to work — the right `line_freq` in
`PreprocessConfig` (60 Hz in the Americas, 50 Hz in most of Europe and Asia).

**Before you load real patient data**, read `LIMITATIONS.md`. This is research
code with no access controls, no audit log and no validation. De-identify
first, work under your own ethics approval, and keep clinical decisions with
the clinical team.
