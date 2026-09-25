# `site/` — the project website

Published at **https://berdakh.github.io/onset-hfo/** by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) on every push to
`master` that touches this directory.

* `index.html` — the whole page. Self-contained: inline CSS, no JavaScript, no
  external stylesheet or font. Edit it directly.
* `img/` — copies of the figures in `onset-hfo/docs/img/`, which the pipeline
  generates with `python -m onset_hfo.cli run --figures`. If you regenerate
  them, copy them here too.

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
