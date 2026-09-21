"""Onset-HFO agent: an open-weight LLM that can read the pipeline's results.

The agent has eight read-only tools, must cite the evidence windows it used,
and is checked after every answer: citations must resolve and numbers must
appear in a tool result. It refuses treatment, diagnosis and other-patient
questions before the model is even called.

    from onset_hfo.store import ResultStore
    from onset_agent import OnsetAgent, make_backend

    store = ResultStore("artifacts/results/sub-pt01_ictal_run-01")
    agent = OnsetAgent(store, make_backend("ollama", model="qwen2.5:7b-instruct"))
    print(agent.ask("Which channels have the highest ripple rate?"))
"""

from onset_agent.agent import AgentAnswer, OnsetAgent  # noqa: F401
from onset_agent.backends import make_backend  # noqa: F401
from onset_agent.tools import TOOLS, tool_schemas  # noqa: F401

__all__ = ["OnsetAgent", "AgentAnswer", "make_backend", "TOOLS", "tool_schemas"]
