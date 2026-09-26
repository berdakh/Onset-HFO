# HFO / epileptiform evidence report -- sub-pt01 (ictal, run 01)

*Report v1, pipeline 0.1.0, generated 2026-09-20T19:59:06+00:00.*

> Decision support for research use. This report presents detector output and the
> signal windows behind it. It contains no diagnosis and no treatment recommendation.

## Summary

2 detectors ran on sub-pt01 (ictal, run 01), 60 s of recording, 98 channels. Highest event rate: ATT5-ATT6 (rms 70.0/min, line_length 90.0/min). 7 of the leading channels are ranked very differently by the two detectors. This report presents evidence windows and rates; it contains no diagnosis and no recommendation.

## Recording

- source: `openneuro:ds003029`
- 98 channels analysed, 1000.0 Hz, 50.0-110.0 s of the original recording
- clinician-marked seizure: 75.95-161.82 s

## Findings

### ATT5-ATT6
- **rms**: 70.0 events/min (95% CI 54.5-87.4, n=70), rank 5
- **line_length**: 90.0 events/min (95% CI 72.3-109.6, n=90), rank 4
- interictal discharges: 8.0/min; 9% of its HFOs coincide with one
- evidence windows: ATT5-ATT6 101.149-101.320 s [sub-pt01|ATT5-ATT6|line_length|101.149]; ATT5-ATT6 101.213-101.295 s [sub-pt01|ATT5-ATT6|rms|101.213]; ATT5-ATT6 101.918-102.034 s [sub-pt01|ATT5-ATT6|line_length|101.918]; ATT5-ATT6 101.932-102.032 s [sub-pt01|ATT5-ATT6|rms|101.932]; ATT5-ATT6 104.399-104.551 s [sub-pt01|ATT5-ATT6|line_length|104.399]; ATT5-ATT6 104.426-104.529 s [sub-pt01|ATT5-ATT6|rms|104.426]

### PD3-PD4
- **rms**: 69.0 events/min (95% CI 53.7-86.3, n=69), rank 7
- **line_length**: 93.0 events/min (95% CI 75.0-112.9, n=93), rank 3
- interictal discharges: 38.0/min; 23% of its HFOs coincide with one
- evidence windows: PD3-PD4 95.528-95.555 s [sub-pt01|PD3-PD4|rms|95.528]; PD3-PD4 95.528-95.567 s [sub-pt01|PD3-PD4|line_length|95.528]; PD3-PD4 106.889-106.959 s [sub-pt01|PD3-PD4|rms|106.889]; PD3-PD4 106.892-106.970 s [sub-pt01|PD3-PD4|line_length|106.892]; PD3-PD4 108.409-108.475 s [sub-pt01|PD3-PD4|rms|108.409]; PD3-PD4 109.703-109.797 s [sub-pt01|PD3-PD4|line_length|109.703]

## Where the detectors disagree

- **PST2-PST3** -- rms: 1, line_length: 7. the two detectors' ranks differ by 5 places; the report states both and resolves neither
  - evidence: PST2-PST3 104.230-104.336 s [sub-pt01|PST2-PST3|line_length|104.230]; PST2-PST3 104.239-104.331 s [sub-pt01|PST2-PST3|rms|104.239]
- **ATT7-ATT8** -- rms: 2, line_length: 17. the two detectors' ranks differ by 15 places; the report states both and resolves neither
  - evidence: ATT7-ATT8 100.350-100.468 s [sub-pt01|ATT7-ATT8|rms|100.350]; ATT7-ATT8 100.354-100.469 s [sub-pt01|ATT7-ATT8|line_length|100.354]
- **ATT6-ATT7** -- rms: 3, line_length: 13. the two detectors' ranks differ by 11 places; the report states both and resolves neither
  - evidence: ATT6-ATT7 100.335-100.463 s [sub-pt01|ATT6-ATT7|rms|100.335]; ATT6-ATT7 101.173-101.314 s [sub-pt01|ATT6-ATT7|rms|101.173]
- **AST2-AST3** -- rms: 4, line_length: 9. the two detectors' ranks differ by 5 places; the report states both and resolves neither
  - evidence: AST2-AST3 102.174-102.316 s [sub-pt01|AST2-AST3|line_length|102.174]; AST2-AST3 102.227-102.317 s [sub-pt01|AST2-AST3|rms|102.227]
- **AD1-AD2** -- rms: 11, line_length: 1. the two detectors' ranks differ by 10 places; the report states both and resolves neither
  - evidence: AD1-AD2 76.957-77.013 s [sub-pt01|AD1-AD2|line_length|76.957]; AD1-AD2 76.958-77.012 s [sub-pt01|AD1-AD2|rms|76.958]
- **AD3-AD4** -- rms: 13, line_length: 2. the two detectors' ranks differ by 11 places; the report states both and resolves neither
  - evidence: AD3-AD4 95.512-95.655 s [sub-pt01|AD3-AD4|line_length|95.512]; AD3-AD4 95.525-95.560 s [sub-pt01|AD3-AD4|rms|95.525]
- **AST1-AST2** -- rms: 18, line_length: 5. the two detectors' ranks differ by 13 places; the report states both and resolves neither
  - evidence: AST1-AST2 96.766-96.848 s [sub-pt01|AST1-AST2|line_length|96.766]; AST1-AST2 98.677-98.792 s [sub-pt01|AST1-AST2|rms|98.677]

## Rate before and during the marked seizure

| channel | before (/min) | during (/min) | ratio |
|---|---|---|---|
| PST2-PST3 | 0.0 | 146.3 | n/a |
| ATT7-ATT8 | 0.0 | 144.5 | n/a |
| ATT6-ATT7 | 0.0 | 144.5 | n/a |
| AST2-AST3 | 0.0 | 141.0 | n/a |
| ATT5-ATT6 | 0.0 | 123.3 | n/a |

## Data quality

- slice 50-110 s of the original recording; all times reported by this pipeline are in original-recording seconds
- 11 channels flagged bad in channels.tsv (white matter, CSF, outside brain, or noisy) and excluded
- rms: 1732 candidate events, 183 rejected by artifact validation (in-band spectral peak only 3.7 dB above the 1/f background (< 4 dB): 13, in-band spectral peak only 3.6 dB above the 1/f background (< 4 dB): 11, in-band spectral peak only 3.2 dB above the 1/f background (< 4 dB): 10, in-band spectral peak only 3.5 dB above the 1/f background (< 4 dB): 10, in-band spectral peak only 3.8 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 3.3 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 2.4 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 3.0 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 3.4 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 3.9 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 1.3 dB above the 1/f background (< 4 dB): 6, in-band spectral peak only 2.9 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 3.1 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 2.2 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 2.6 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 0.3 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 0.2 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 1.4 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 2.3 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only -0.8 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 2.8 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 1.7 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 1.9 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 2.7 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 1.0 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 0.4 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only -0.5 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 0.9 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 0.6 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 0.5 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only -0.7 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.8 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 2.1 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.6 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 2.5 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.2 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.1 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.5 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 4.0 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 2.0 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only -1.9 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.3 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.6 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.6 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.0 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -2.4 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only 0.8 dB above the 1/f background (< 4 dB): 1, only 1.9 cycles (< 2): 1, only 2.0 cycles (< 2): 1, in-band spectral peak only -0.4 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.2 dB above the 1/f background (< 4 dB): 1)
- line_length: 2387 candidate events, 199 rejected by artifact validation (in-band spectral peak only 3.9 dB above the 1/f background (< 4 dB): 12, in-band spectral peak only 3.5 dB above the 1/f background (< 4 dB): 11, in-band spectral peak only 3.8 dB above the 1/f background (< 4 dB): 10, in-band spectral peak only 3.3 dB above the 1/f background (< 4 dB): 10, in-band spectral peak only 3.0 dB above the 1/f background (< 4 dB): 10, in-band spectral peak only 3.4 dB above the 1/f background (< 4 dB): 9, in-band spectral peak only 3.6 dB above the 1/f background (< 4 dB): 9, in-band spectral peak only 3.2 dB above the 1/f background (< 4 dB): 8, in-band spectral peak only 2.7 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 2.0 dB above the 1/f background (< 4 dB): 7, in-band spectral peak only 2.8 dB above the 1/f background (< 4 dB): 6, in-band spectral peak only 2.9 dB above the 1/f background (< 4 dB): 6, in-band spectral peak only 2.1 dB above the 1/f background (< 4 dB): 6, in-band spectral peak only 1.5 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 4.0 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 3.7 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 2.4 dB above the 1/f background (< 4 dB): 5, in-band spectral peak only 1.8 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 1.2 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 1.3 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 2.6 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 3.1 dB above the 1/f background (< 4 dB): 4, in-band spectral peak only 0.6 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 2.3 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 1.6 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 1.7 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 1.4 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only 0.4 dB above the 1/f background (< 4 dB): 3, in-band spectral peak only -0.8 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 0.8 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only -0.6 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.9 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 0.5 dB above the 1/f background (< 4 dB): 2, only 1.9 cycles (< 2): 2, in-band spectral peak only 0.3 dB above the 1/f background (< 4 dB): 2, in-band spectral peak only 1.0 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.1 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.9 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.4 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.3 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.1 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only 0.9 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.0 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.3 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only 0.7 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.2 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -2.3 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only 1.1 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -0.5 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.5 dB above the 1/f background (< 4 dB): 1, in-band spectral peak only -1.6 dB above the 1/f background (< 4 dB): 1)
- rates are events per minute over the analysed window; confidence intervals assume a Poisson process

## Method

- kept 89 intracranial channels; dropped 9 non-brain channels (DC/trigger/ECG/misc)
- dropped 4 channels flagged bad by the dataset: G5, G6, RQ1, RQ2
- high-pass 1 Hz (zero-phase FIR)
- notch 60 Hz + harmonics (60, 120, 180, 240, 300, 360, 420 Hz, 2 Hz wide)
- bipolar montage: 71 pairs of neighbouring contacts
- detectors: rms, line_length (see docs/METHODS.md)
- configuration: {"preprocess": {"line_freq": 60.0, "notch": true, "bipolar": true, "highpass": 1.0, "drop_bads": true}, "rms": {"band": [80.0, 250.0], "rms_window_ms": 3.0, "threshold_sd": 5.0, "extend_sd": 2.0, "peak_threshold_sd": 2.0, "min_peaks": 6, "min_duration_ms": 6.0, "max_duration_ms": 200.0, "merge_gap_ms": 10.0, "baseline": "robust"}, "line_length": {"band": [80.0, 250.0], "rms_window_ms": 3.0, "threshold_sd": 3.0, "extend_sd": 2.0, "peak_threshold_sd": 2.0, "min_peaks": 6, "min_duration_ms": 6.0, "max_duration_ms": 200.0, "merge_gap_ms": 10.0, "baseline": "robust"}, "spikes": {"band": [5.0, 60.0], "threshold_sd": 6.0, "slope_sd": 5.0, "min_duration_ms": 10.0, "max_duration_ms": 200.0, "refractory_ms": 200.0, "max_raw_jump_sd": 10.0}, "validation": {"min_peak_prominence_db": 4.0, "max_low_band_sd": 8.0, "min_cycles": 2.0}, "rate_window_s": 60.0, "top_k": 5, "disagreement_ranks": 5}

## Limitations

- prototype: thresholds were chosen from the literature and checked on synthetic data, not tuned or validated on a labelled clinical cohort
- no HFO ground truth exists for this public recording, so precision and recall are not reported here; they are measured on synthetic data in docs/EVALUATION.md
- event rate is not a diagnosis: physiological ripples occur in healthy tissue, particularly in mesial temporal and occipital regions
- a single short window of a single patient; nothing here generalises
- bipolar pairs are formed from consecutive contact numbers, which on a grid is not always spatial adjacency

## Data citation

Li A, Inati S, Zaghloul K, Crone N, Anderson W, Johnson E, Cajigas I, Brusko D, Jagid J, Claudio A, Kanner A, Hopp J, Chen S, Haagensen J, Sarma S. Epilepsy-iEEG-Multicenter-Dataset. OpenNeuro (2021). doi:10.18112/openneuro.ds003029.v1.0.3 -- the archive asks that work using it also cite 'Neural fragility as an EEG marker of the seizure onset zone', doi:10.1101/862797 (Nature Neuroscience, 2023).
