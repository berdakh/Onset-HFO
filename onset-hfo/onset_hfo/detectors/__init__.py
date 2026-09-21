"""The detectors.

Two HFO detectors (deliberately simple, deliberately different) and one
interictal-spike detector:

* :func:`detect_rms` -- energy-based, after Staba et al. (2002).
* :func:`detect_line_length` -- waveform-length based, after Gardner et al. (2007).
* :func:`detect_spikes` -- amplitude + sharpness, an interictal epileptiform
  discharge (IED) detector.

They share the primitives in :mod:`onset_hfo.detectors.base`, so that the only
real difference between the two HFO detectors is the feature they threshold.
That is what makes "the detectors disagree here" a meaningful statement.
"""

from onset_hfo.detectors.base import Event, events_to_frame, frame_to_events  # noqa: F401
from onset_hfo.detectors.line_length import detect_line_length  # noqa: F401
from onset_hfo.detectors.rms import detect_rms  # noqa: F401
from onset_hfo.detectors.spike import detect_spikes  # noqa: F401

DETECTORS = {"rms": detect_rms, "line_length": detect_line_length, "spike": detect_spikes}

__all__ = ["Event", "events_to_frame", "frame_to_events", "detect_rms",
           "detect_line_length", "detect_spikes", "DETECTORS"]
