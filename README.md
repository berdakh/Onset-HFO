# BCI-GAN project
In this project, the application of GANs to augment EEG-ERP data is investigated.

To train models, use files starting with 'train_'. Select and hardcode parameters first:

<pre><code>python train_subject_specific.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_subject_independent.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_gan_test.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_subject_specific.py --train_type {subject-specific, subject-independent}
</code></pre>

---

## Onset-HFO — HFO / epileptiform detection prototype (`onset-hfo/`)

A separate, self-contained prototype lives in [`onset-hfo/`](onset-hfo/): it
detects high-frequency oscillations (ripples, 80–250 Hz) and interictal
epileptiform discharges in **public** intracranial EEG (OpenNeuro `ds003029`,
CC0), and ships an evidence-only agent built on an **open-weight** language
model that can read the results, must cite them, and refuses clinical
questions.

* Three Colab notebooks that run with no setup — detection quickstart, the
  agent, and a validation benchmark against known ground truth.
* Documentation for every part of the system in
  [`onset-hfo/docs/`](onset-hfo/docs/), including what it must not be used for.
* 58 offline tests; `python -m onset_hfo.cli run --synthetic` runs the whole
  pipeline in a few seconds with no download.

Start at [`onset-hfo/README.md`](onset-hfo/README.md). It shares nothing with
the GAN code above except this repository.
