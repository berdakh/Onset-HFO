"""Ask the agent something about a saved analysis.

    # offline, no model required (deterministic scripted policy)
    python -m onset_agent.cli --results artifacts/results/sim-01_simulated_run-01 \
        --question "which channels have the highest ripple rate?"

    # with a local open-weight model
    ollama pull qwen2.5:7b-instruct && ollama serve
    python -m onset_agent.cli --results <dir> --backend ollama --question "..."

    # the demonstration set, including the questions that must be refused
    python -m onset_agent.cli --results <dir> --demo

    # interactive
    python -m onset_agent.cli --results <dir> --backend ollama --chat
"""

from __future__ import annotations

import argparse
import json
import sys

from onset_agent.agent import OnsetAgent
from onset_agent.backends import make_backend
from onset_agent.prompts import EXAMPLE_QUESTIONS
from onset_hfo.store import ResultStore


def _print(answer, show_trace: bool = False) -> None:
    mark = "refused" if answer.refused else "answer"
    print(f"\nQ: {answer.question}\n[{mark}] {answer.text}")
    if answer.evidence_ids:
        print("  cites: " + ", ".join(answer.evidence_ids))
    if answer.tools_called:
        print("  tools: " + " -> ".join(answer.tools_called))
    if answer.refused and answer.reason:
        print(f"  reason: {answer.reason}")
    if show_trace:
        print("  trace: " + json.dumps(answer.trace, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="onset-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="a results directory from onset_hfo.cli run")
    p.add_argument("--backend", default="scripted",
                   choices=["scripted", "ollama", "openai_compat", "transformers"])
    p.add_argument("--model", default=None, help="model name/id for the chosen backend")
    p.add_argument("--base-url", default=None, help="server URL for ollama/openai_compat")
    p.add_argument("--question", "-q", default=None)
    p.add_argument("--demo", action="store_true", help="run the example question set")
    p.add_argument("--chat", action="store_true", help="interactive prompt")
    p.add_argument("--trace", action="store_true", help="print the full tool trace")
    p.add_argument("--max-steps", type=int, default=6)
    args = p.parse_args(argv)

    store = ResultStore(args.results)
    backend = make_backend(args.backend, model=args.model, base_url=args.base_url)
    agent = OnsetAgent(store, backend, max_steps=args.max_steps)

    print(f"[onset-agent] analysis: {store.subject} ({store.metadata().get('source')}), "
          f"{len(store.channels())} channels")
    print(f"[onset-agent] backend: {backend.describe()}")
    if not backend.is_language_model:
        print("[onset-agent] NOTE: the scripted backend is not a language model. "
              "Use --backend ollama or transformers for the real thing.")

    if args.demo:
        for answer in agent.ask_many(EXAMPLE_QUESTIONS):
            _print(answer, args.trace)
        return 0
    if args.question:
        _print(agent.ask(args.question), args.trace)
        return 0
    if args.chat:
        print("[onset-agent] type a question, or 'quit'.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"quit", "exit", "q"}:
                return 0
            if question:
                _print(agent.ask(question), args.trace)
    p.error("give --question, --demo or --chat")
    return 2


if __name__ == "__main__":
    sys.exit(main())
