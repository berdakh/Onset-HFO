"""Line-length HFO detector -- after Gardner et al., Clin Neurophysiol 118:1134 (2007).

Line length is the mean absolute sample-to-sample difference in a sliding
window. It grows with both amplitude *and* frequency, so it reacts to fast
low-amplitude oscillations that an energy detector can miss. The price is that
it also reacts to sharp edges, which is why its threshold is lower (3 robust
SDs by default) and why the artifact validation stage matters even more here.

Running this detector next to :func:`~onset_hfo.detectors.rms.detect_rms` is
the point: where two simple detectors built on the same preprocessing and the
same band disagree, the prototype says so instead of averaging the
disagreement away.
"""

from __future__ import annotations

from onset_hfo.config import DetectorConfig
from onset_hfo.detectors.base import Event, sliding_line_length
from onset_hfo.detectors.engine import detect_with_feature
from onset_hfo.preprocess import Prepared

__all__ = ["detect_line_length"]


def detect_line_length(prep: Prepared, cfg: DetectorConfig | None = None, **kwargs) -> list[Event]:
    """Detect ripple-band events by sliding-window line length."""
    cfg = cfg or DetectorConfig(threshold_sd=3.0)
    return detect_with_feature(prep, cfg, lambda x, w: sliding_line_length(x, w),
                               "line_length", **kwargs)
