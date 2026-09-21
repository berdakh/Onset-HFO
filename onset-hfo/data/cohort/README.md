# The cohort feature table

`features.csv.gz` is the output of

```bash
python -m onset_hfo.learn cohort
```

on OpenNeuro `ds003029` — **one row per analysed channel per subject**, 22
subjects, 1466 channels, 301 labelled SOZ. It is committed because rebuilding
it means downloading about 700 MB and twenty minutes of signal processing,
and without it nobody can re-run the modelling half of the repository.

It contains no signal: every column is a per-channel summary the pipeline
already prints in its report, plus the clinician SOZ label from the archive's
own `sourcedata/clinical_data_summary.xlsx`. The source data is CC0; if you
publish anything derived from this, cite the dataset and its paper (the
citation is in every report and in `provenance.json`).

```python
import pandas as pd
features = pd.read_csv("data/cohort/features.csv.gz")
```

| file | what it is |
|---|---|
| `features.csv.gz` | the feature table; `*_z` and `*_rank` columns are normalised within subject |
| `plan.csv` | what the runner decided to analyse, and the seizure marker it found |
| `failures.csv` | the 9 subjects it could not analyse, with the reason for each |
| `evaluation.csv` | the table in `docs/LOCALIZATION.md` §2 |
| `summary.json` | counts per site |

**Read `failures.csv`.** Four subjects were excluded because their recordings
sample at 250 Hz, where an 80–250 Hz ripple is not measurable at all, and four
because the archive's signal file is shorter than its own marked seizure time.
A cohort of 22 rather than 31 is the honest denominator for every number
computed from this table.

This is derived from a fixed pipeline version. Re-run the command above rather
than trusting it if you have changed a detector, a threshold or the
preprocessing.
