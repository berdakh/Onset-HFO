"""Onset-HFO agent: open-weight language models driving, and being checked by, code.

Two halves, which can be used separately.

**The evidence assistant** (:class:`~onset_agent.agent.OnsetAgent`) answers
questions about a *saved* analysis through eight read-only tools. It must cite
the evidence windows it used, every number it states is checked against what
the tools returned, and treatment, diagnosis and other-patient questions are
refused before the model is called at all::

    from onset_hfo.store import ResultStore
    from onset_agent import OnsetAgent, make_backend

    store = ResultStore("artifacts/results/sub-pt01_ictal_run-01")
    agent = OnsetAgent(store, make_backend("ollama", model="qwen2.5:7b-instruct"))
    print(agent.ask("Which channels have the highest ripple rate?"))

**The orchestrator** (:mod:`onset_agent.planner`) lets the model choose and
*parameterise* the analyses instead of reading precomputed ones. Every tool
call runs the real pipeline, is recorded with a run id in an evidence store,
and every claim in the final report must resolve back to one. Four rungs of
increasing model control (S0-S3) share the identical analyzers, so any
difference between them is attributable to orchestration::

    from onset_agent import AnalysisSession, Rung, run_rung
    from onset_hfo.datasets import fetch_slice

    session = AnalysisSession(fetch_slice("sub-pt01", "ictal", "01", 50, 110))
    fixed   = run_rung(session, Rung.S0)   # fixed pipeline, no model
    planned = run_rung(session, Rung.S2)   # the model re-plans as results arrive

See ``docs/ORCHESTRATION.md``.
"""

from onset_agent.agent import AgentAnswer, OnsetAgent  # noqa: F401
from onset_agent.analysis import AnalysisSession, build_registry  # noqa: F401
from onset_agent.backends import make_backend  # noqa: F401
from onset_agent.contract import CONTRACT_VERSION, ToolRun  # noqa: F401
from onset_agent.evidence import EvidenceStore  # noqa: F401
from onset_agent.planner import PlannerResult, Rung, rank_channels, run_rung  # noqa: F401
from onset_agent.tools import TOOLS, tool_schemas  # noqa: F401
from onset_agent.verifier import DeterministicVerifier, LLMVerifier  # noqa: F401

__all__ = [
    # the evidence assistant
    "OnsetAgent", "AgentAnswer", "make_backend", "TOOLS", "tool_schemas",
    # the orchestrator
    "AnalysisSession", "build_registry", "Rung", "run_rung", "PlannerResult",
    "rank_channels", "EvidenceStore", "ToolRun", "CONTRACT_VERSION",
    "DeterministicVerifier", "LLMVerifier",
]
