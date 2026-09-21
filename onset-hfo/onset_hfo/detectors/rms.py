"""Energy (RMS) HFO detector -- after Staba et al., J Neurophysiol 88:1743 (2002).

The original recipe, and what this implementation keeps:

* band-pass 80-500 Hz (here: the configured ripple band, 80-250 Hz by default,
  because the example recording is sampled at 1000 Hz);
* RMS in a 3 ms sliding window;
* threshold at 5 standard deviations of that channel's own RMS;
* keep crossings lasting at least 6 ms;
* require at least 6 rectified peaks above a secondary threshold inside the
  event (a ripple has about two rectified peaks per cycle, so six peaks is
  roughly three cycles).

What this implementation changes, on purpose:

* the "standard deviation" is a median-absolute-deviation estimate by default
  (``baseline="robust"``). A channel full of HFOs inflates its own SD and
  hides its own events; the MAD does not care. Set ``baseline="sd"`` to get
  the literal 2002 behaviour.
* every event is passed to :mod:`onset_hfo.validate` afterwards, which is
  where filter ringing from sharp transients is removed.
"""

from __future__ import annotations

from onset_hfo.config import DetectorConfig
from onset_hfo.detectors.base import Event, sliding_rms
from onset_hfo.detectors.engine import detect_with_feature
from onset_hfo.preprocess import Prepared

__all__ = ["detect_rms"]


def detect_rms(prep: Prepared, cfg: DetectorConfig | None = None, **kwargs) -> list[Event]:
    """Detect ripple-band events by sliding-window RMS energy."""
    cfg = cfg or DetectorConfig()
    return detect_with_feature(prep, cfg, lambda x, w: sliding_rms(x, w), "rms", **kwargs)
