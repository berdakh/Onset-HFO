"""Shared fixtures. Everything here is offline: no download, no model, no GPU."""

from __future__ import annotations

import pytest

from onset_hfo.pipeline import run_pipeline
from onset_hfo.preprocess import prepare
from onset_hfo.synthetic import make_synthetic_recording


@pytest.fixture(scope="session")
def recording():
    """A short labelled synthetic recording (seed fixed, so tests are stable)."""
    return make_synthetic_recording(duration_s=30, seed=7, verbose=False)


@pytest.fixture(scope="session")
def prepared(recording):
    return prepare(recording, verbose=False)


@pytest.fixture(scope="session")
def result(recording):
    return run_pipeline(recording, verbose=False)


@pytest.fixture(scope="session")
def store(result, tmp_path_factory):
    from onset_hfo.store import ResultStore

    return ResultStore(result.save(tmp_path_factory.mktemp("results")))


@pytest.fixture(scope="session")
def session(recording):
    """A live analysis session over the synthetic recording.

    Session-scoped so the montage and band-pass are computed once; tests that
    care about cost call ``reset_memo`` themselves.
    """
    from onset_agent.analysis import AnalysisSession

    return AnalysisSession(recording, verbose=False)


@pytest.fixture
def registry(session):
    from onset_agent.analysis import build_registry

    session.reset_memo()
    return build_registry(session)
