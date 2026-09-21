# Onset-HFO

**Onset-HFO: detecting high-frequency oscillations and interictal epileptiform
discharges in public intracranial EEG, with an evidence-only agent on an
open-weight model. Research prototype — not a medical device.**

Everything lives in [`onset-hfo/`](onset-hfo/). Start there:
[**onset-hfo/README.md**](onset-hfo/README.md).

| | Notebook (runs in Colab, no setup) | What it shows |
|---|---|---|
| 1 | [HFO detection quickstart](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/01_hfo_detection_quickstart.ipynb) | real public iEEG → detections → figures → a cited report |
| 2 | [Agentic analysis](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/02_agentic_analysis.ipynb) | an open-weight model answering questions about those results, with citations, refusals and guards |
| 3 | [Validation and benchmark](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/03_validation_and_benchmark.ipynb) | precision/recall against known truth, threshold curves, what each check buys |

```bash
git clone https://github.com/berdakh/onset-hfo.git
cd onset-hfo/onset-hfo
pip install -e ".[dev]"

python -m onset_hfo.cli run --synthetic --figures   # offline, ~10 seconds
python -m onset_hfo.cli run --subject sub-pt01 --start 50 --stop 110
python -m onset_agent.cli --results artifacts/results/sub-pt01_ictal_run-01 --demo
pytest -q                                           # 58 tests, offline
```

Part of the [Onset](https://berdakh.github.io/onset/) project —
Brain–Machine Interfaces Lab, Nazarbayev University. MIT licence.
Full documentation in [`onset-hfo/docs/`](onset-hfo/docs/), including
[what this must not be used for](onset-hfo/docs/LIMITATIONS.md).

---

## About the earlier contents of this repository

This repository previously held **BCI-GAN**, an investigation into using GANs
(DCGAN, WGAN-GP, VAE) to augment EEG-ERP data. Those files were removed from
the working tree; **nothing is lost** — they remain in the git history and can
be restored at any time:

```bash
# see them
git show 38331a8 --stat -- '*.py'

# restore one, or all of them
git checkout 38331a8 -- dcgan.py
git checkout 38331a8 -- cnn.py data_import.py dcgan.py gan_test.py vae.py \
    wgan_gp.py train_gan_test.py train_subject_independent.py \
    train_subject_specific.py train_without_gans.py
```

Commit `38331a8` is the last one in which they are present. Their original
usage was `python train_subject_specific.py --gan_type {dcgan, wgan_gp, vae}`
and the equivalent `train_subject_independent.py` / `train_gan_test.py`.
