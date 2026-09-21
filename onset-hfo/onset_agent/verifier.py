"""Verification: every claim resolves to a run id, or it is struck.

The rule is simple and absolute: **a sentence that states a number is
admissible only if that number appears in some tool's output.** There is no
softening, no hedging, no "approximately". A sentence that fails is removed
from the report and recorded, with its reason, in the audit trail.

Two verifiers, and the point is to have both
--------------------------------------------
:class:`DeterministicVerifier` resolves numbers against the evidence store by
arithmetic. It is exact, instant, reproducible, and cannot be talked out of a
verdict. :class:`LLMVerifier` is a second model instance that reads the draft
and the evidence and judges each sentence, as the thesis proposal specifies.

Running only one of them would answer the wrong question. The research
question is not "does verification remove hallucinations" -- it does, by
construction -- but **what verification costs**: a verifier also strikes true
statements it cannot resolve, and a report with the hallucinations and half
the findings removed is not obviously better than one with neither.
:func:`compare_verifiers` measures exactly that, sentence by sentence.

What counts as a claim
----------------------
A sentence containing at least one number that is not part of a channel name,
an evidence id or a bracketed run id. Prose about method or limitations
("rates come from one short window") carries no number and is not a claim, so
it is never struck -- verification must not quietly delete the caveats.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from onset_agent.evidence import EvidenceStore

__all__ = ["VerificationResult", "DeterministicVerifier", "LLMVerifier",
           "compare_verifiers", "split_sentences", "claim_numbers"]

#: Split on sentence punctuation followed by a capital. A decimal point is
#: followed by a digit, so "14.2 ripples/min" survives intact.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
#: A run id the model cited inline: "... 14.2 ripples/min [hfo_003]."
_RUN_CITE = re.compile(r"\[([a-z_]+_\d+)\]")
#: Tokens whose digits are not measurements.
_CHANNEL_TOKEN = re.compile(r"\b[A-Za-z]{1,6}'?\d{1,3}(?:-[A-Za-z]{1,6}'?\d{1,3})?\b")
_EVIDENCE_TOKEN = re.compile(r"\b\S+\|\S+\|\S+\|\S+\b")
_NUMBER = re.compile(r"(?<![\d.\w])(-?\d+(?:\.\d+)?)(?![\d.]*\w)")
#: A small whole number with no unit is counting something the writer may
#: legitimately count ("the top 3 channels", "two detectors").
MAX_FREE_INTEGER = 20


def split_sentences(text: str) -> list[str]:
    """Split a report into sentences, keeping decimals and citations intact."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def claim_numbers(sentence: str) -> list[float]:
    """The numbers in a sentence that have to be traceable to a tool run."""
    stripped = _RUN_CITE.sub(" ", sentence)
    stripped = _EVIDENCE_TOKEN.sub(" ", stripped)
    stripped = _CHANNEL_TOKEN.sub(" ", stripped)
    out = []
    for raw in _NUMBER.findall(stripped):
        value = float(raw)
        if value.is_integer() and 0 <= value <= MAX_FREE_INTEGER:
            continue
        out.append(round(value, 4))
    return out


@dataclass
class VerificationResult:
    """What a verifier decided, and the numbers a reviewer will want."""

    verifier: str
    kept_text: str = ""
    kept: list[str] = field(default_factory=list)
    struck: list[dict] = field(default_factory=list)
    supported: list[dict] = field(default_factory=list)
    n_sentences: int = 0
    n_claims: int = 0

    @property
    def n_struck(self) -> int:
        return len(self.struck)

    @property
    def unsupported_claim_rate(self) -> float:
        """Fraction of claims that could not be resolved to a run id.

        The first number any reviewer asks for. Zero means every number in the
        surviving report came from a measurement.
        """
        return (self.n_struck / self.n_claims) if self.n_claims else 0.0

    @property
    def provenance_coverage(self) -> float:
        """Fraction of claims that resolved to at least one run id."""
        return 1.0 - self.unsupported_claim_rate

    def missing_evidence(self) -> list[str]:
        """What the planner would have to measure to rescue the struck claims."""
        return sorted({s.get("reason", "") for s in self.struck})

    def as_dict(self) -> dict:
        return {"verifier": self.verifier, "n_sentences": self.n_sentences,
                "n_claims": self.n_claims, "n_struck": self.n_struck,
                "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
                "provenance_coverage": round(self.provenance_coverage, 4),
                "struck": self.struck}


class DeterministicVerifier:
    """Resolve every claimed number against the evidence store, by arithmetic.

    A sentence survives when each of its numbers appears in some tool output
    (within :data:`~onset_agent.evidence.NUMBER_TOLERANCE`). If the sentence
    also cites a run id inline, that id must exist *and* must be one of the
    runs that actually produced the number -- citing ``hfo_001`` for a figure
    that only ``compare_001`` reported is a miscitation, not a rounding error.
    """

    name = "deterministic"

    def __call__(self, report: str, store: EvidenceStore) -> VerificationResult:
        result = VerificationResult(verifier=self.name)
        sentences = split_sentences(report)
        result.n_sentences = len(sentences)
        kept = []
        for index, sentence in enumerate(sentences):
            numbers = claim_numbers(sentence)
            if not numbers:
                kept.append(sentence)          # prose and caveats are never struck
                continue
            result.n_claims += 1
            cited = _RUN_CITE.findall(sentence)
            resolved: dict[str, list[str]] = {}
            unsupported = []
            for number in numbers:
                run_ids = store.find_number(number)
                if run_ids:
                    resolved[str(number)] = run_ids
                else:
                    unsupported.append(number)
            if unsupported:
                result.struck.append({
                    "index": index, "sentence": sentence,
                    "reason": (f"no tool run produced "
                               f"{', '.join(_fmt(n) for n in unsupported)}"),
                    "unsupported_numbers": unsupported})
                continue
            bad_cite = [c for c in cited if store.get(c) is None]
            if bad_cite:
                result.struck.append({
                    "index": index, "sentence": sentence,
                    "reason": f"cites run id(s) that do not exist: {', '.join(bad_cite)}",
                    "unsupported_numbers": []})
                continue
            supporting = {rid for ids in resolved.values() for rid in ids}
            miscited = [c for c in cited if c not in supporting]
            if miscited:
                result.struck.append({
                    "index": index, "sentence": sentence,
                    "reason": (f"cites {', '.join(miscited)}, which did not produce the "
                               f"number(s) claimed"),
                    "unsupported_numbers": []})
                continue
            kept.append(sentence)
            result.supported.append({"index": index, "sentence": sentence,
                                     "run_ids": sorted(supporting)})
        result.kept = kept
        result.kept_text = " ".join(kept)
        return result


VERIFIER_PROMPT = """You are checking a draft research report against the analyses that \
produced it. For each numbered sentence, decide whether every number in it appears in the \
evidence below.

EVIDENCE
{evidence}

DRAFT
{draft}

A sentence is supported only if each of its numbers can be found in the evidence. A sentence \
with no numbers in it is supported. Do not correct anything and do not add anything.

Reply with ONE JSON object and nothing else:
{{"verdicts": [{{"index": 0, "supported": true, "run_ids": ["hfo_001"], "reason": ""}}, ...]}}"""


class LLMVerifier:
    """A second model instance that judges each sentence against the evidence.

    This is the verifier the thesis proposal describes. It can do something
    the deterministic one cannot -- catch a sentence whose numbers are all
    real but whose *claim* about them is wrong ("the rate doubled" when it
    halved) -- and it fails in ways the deterministic one cannot, which is why
    both are kept and :func:`compare_verifiers` reports the difference.

    A malformed or missing verdict is treated as "supported". The alternative
    -- striking a sentence because the verifier's JSON was broken -- would
    make report quality a function of decoding luck.
    """

    name = "llm"

    def __init__(self, backend, max_chars: int = 6000):
        self.backend = backend
        self.max_chars = max_chars

    def __call__(self, report: str, store: EvidenceStore) -> VerificationResult:
        from onset_agent.backends import extract_json_object

        result = VerificationResult(verifier=f"{self.name}:{self.backend.name}")
        sentences = split_sentences(report)
        result.n_sentences = len(sentences)
        result.n_claims = sum(1 for s in sentences if claim_numbers(s))
        if not sentences:
            return result

        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        prompt = VERIFIER_PROMPT.format(evidence=store.digest()[:self.max_chars],
                                        draft=numbered)
        message = self.backend.chat([{"role": "user", "content": prompt}], [])
        parsed = extract_json_object(message.content or "") or {}
        verdicts = {int(v["index"]): v for v in parsed.get("verdicts", [])
                    if isinstance(v, dict) and str(v.get("index", "")).isdigit()}

        kept = []
        for index, sentence in enumerate(sentences):
            verdict = verdicts.get(index)
            if verdict is None or verdict.get("supported", True):
                kept.append(sentence)
                if claim_numbers(sentence):
                    result.supported.append({
                        "index": index, "sentence": sentence,
                        "run_ids": [str(r) for r in (verdict or {}).get("run_ids", [])]})
                continue
            result.struck.append({"index": index, "sentence": sentence,
                                  "reason": str(verdict.get("reason") or
                                                "the verifier could not resolve it"),
                                  "unsupported_numbers": []})
        result.kept = kept
        result.kept_text = " ".join(kept)
        return result


def compare_verifiers(report: str, store: EvidenceStore, llm_backend) -> dict:
    """Run both verifiers on one draft and measure where they differ.

    This is the RQ2 measurement. The interesting cells are the off-diagonal
    ones:

    * **struck by the model only** -- the language-model verifier removing
      statements whose numbers are demonstrably in the evidence. Every one of
      these is a true statement lost, and their count is the coverage cost of
      using a model as the gate.
    * **struck by arithmetic only** -- sentences the model waved through whose
      numbers no tool produced. Every one of these is a hallucination the
      model verifier missed.

    Neither verifier is treated as ground truth. The deterministic one is
    ground truth *about arithmetic*, which is a smaller claim and the only one
    that can be made without an annotator.
    """
    strict = DeterministicVerifier()(report, store)
    loose = LLMVerifier(llm_backend)(report, store)
    strict_struck = {s["index"] for s in strict.struck}
    loose_struck = {s["index"] for s in loose.struck}
    sentences = split_sentences(report)
    claims = {i for i, s in enumerate(sentences) if claim_numbers(s)}
    return {
        "n_sentences": len(sentences),
        "n_claims": len(claims),
        "deterministic": strict.as_dict(),
        "llm": loose.as_dict(),
        "both_struck": sorted(strict_struck & loose_struck),
        "struck_by_llm_only": sorted(loose_struck - strict_struck),
        "struck_by_arithmetic_only": sorted(strict_struck - loose_struck),
        "agreement": round(
            1.0 - len(strict_struck ^ loose_struck) / len(claims), 4) if claims else 1.0,
        "coverage_cost": len(loose_struck - strict_struck),
        "hallucinations_missed_by_llm": len(strict_struck - loose_struck),
        "reading": ("struck_by_llm_only counts true statements the model verifier removed; "
                    "struck_by_arithmetic_only counts unsupported numbers it let through"),
    }


def _fmt(number: float) -> str:
    return f"{number:g}"


def verifier_from_backend(backend=None):
    """The deterministic verifier, or the model one when a backend is given."""
    return DeterministicVerifier() if backend is None else LLMVerifier(backend)


def repair_request(result: VerificationResult) -> str:
    """The message the planner is given when claims were struck.

    Phrased as a request for measurement rather than a request for rewording,
    because the fix for an unsupported number is to go and measure it, not to
    say it more carefully.
    """
    if not result.struck:
        return ""
    reasons = "; ".join(s["reason"] for s in result.struck[:4])
    return ("The following claims were removed because they could not be traced to a tool "
            f"run: {reasons}. Run the analyses that would produce those numbers, then the "
            "report will be drafted again.")


VERIFIER_JSON_EXAMPLE = json.dumps(
    {"verdicts": [{"index": 0, "supported": True, "run_ids": ["hfo_001"], "reason": ""}]},
    indent=2)
