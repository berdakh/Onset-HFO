# `site/` — the project website

Published at **https://berdakh.github.io/onset-hfo/** by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) on every push to
`master` that touches this directory.

The workflow copies this directory onto the `gh-pages` branch, which holds
nothing else and is rewritten from scratch each time. Do not edit `gh-pages`
by hand -- the next publish overwrites it. GitHub's own Pages actions were
tried first and could not be used: they ask the API to create the Pages site,
and this repository's workflow token is not permitted to do that.

* `index.html` — the whole page. Self-contained: inline CSS, no JavaScript, no
  external stylesheet or font. Edit it directly.
* `img/` — copies of the figures in `onset-hfo/docs/img/`, which the pipeline
  generates with `python -m onset_hfo.cli run --figures`. If you regenerate
  them, copy them here too.
* `handout.html` — the one-page handout for the expo, described in
  [`onset-hfo/docs/EXPO.md`](../onset-hfo/docs/EXPO.md). Open it and print to
  PDF: its print stylesheet sets A4 and it is laid out to fill exactly one
  page. If you add to it, re-check that it still prints as one page.

Two rules to keep in mind when editing:

1. **Use relative paths** (`img/fig.png`, not `/img/fig.png`). The site is
   served under the `/onset-hfo/` subpath, so an absolute path leaves the site.
2. **Keep the numbers in step with `docs/EVALUATION.md`.** Every figure quoted
   on the page is measured, and a page that drifts from the measurements is
   worse than no page.

To preview locally:

```bash
python -m http.server -d site 8000   # then open http://localhost:8000
```
