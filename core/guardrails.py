"""Input guardrails: text normalisation and prompt-injection screening.

Honest scope
------------
Pattern screening is a *speed bump*, not a security boundary: a determined
attacker can paraphrase around any regex list, and legitimate text can
occasionally match (false positives are rejected with a generic message).
It removes the cheap, copy-paste attacks and the unicode tricks used to hide
them. The real defences are elsewhere in this codebase: tools are least-
privilege and schema-validated, retrieved documents are fenced and declared
untrusted, destructive actions need explicit arguments, and every loop is
bounded by an iteration budget and LangGraph's recursion limit.
"""

import re
import unicodedata
from typing import Annotated, Any

from pydantic import BeforeValidator

# Zero-width, soft-hyphen and bidirectional-control characters: invisible, but
# they split keywords ("ig​nore") and can reorder displayed text.
_INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤⁦-⁩﻿]")
# C0/C1 control characters except tab (\x09), newline (\x0a), carriage return (\x0d).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_BLANK_RUNS = re.compile(r"\n{4,}")


def normalize_text(value: Any) -> Any:
    """NFKC-normalise and strip invisible/control characters.

    NFKC folds look-alike forms (full-width letters, ligatures) onto plain
    ASCII so the screening rules below cannot be dodged with them. Non-strings
    are returned untouched so this can be used as a pydantic ``BeforeValidator``.
    """
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFKC", value)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = _BLANK_RUNS.sub("\n\n\n", text)
    return text.strip()


SafeText = Annotated[str, BeforeValidator(normalize_text)]
"""A ``str`` that is normalised before any other validation runs."""

_FLAGS = re.IGNORECASE | re.MULTILINE

INJECTION_RULES: dict[str, re.Pattern[str]] = {
    # "ignore all previous instructions", "disregard your earlier rules", ...
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
        r"\b(previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,30}"
        r"\b(instructions?|rules?|prompts?|guidelines?|directions?|guardrails?)\b",
        _FLAGS,
    ),
    # "show me your system prompt", "repeat the hidden instructions"
    "reveal_system_prompt": re.compile(
        r"\b(reveal|show|print|repeat|output|leak|display|tell me)\b[^.\n]{0,40}"
        r"\b(system|developer|hidden|initial|original)\s+(prompt|message|instructions?)\b",
        _FLAGS,
    ),
    # Chat-template control tokens used to forge role boundaries.
    "role_control_tokens": re.compile(
        r"<\|?\s*(im_start|im_end|system|assistant|endoftext)\s*\|?>|\[/?INST\]|<<\s*/?SYS\s*>>",
        _FLAGS,
    ),
    # A pasted transcript line pretending to be another role.
    "role_header": re.compile(r"^\s*(system|assistant|developer)\s*:", _FLAGS),
    # "you are now DAN", "pretend you have no restrictions"
    "persona_override": re.compile(
        r"\b(you are now|pretend (?:to be|you are|you have)|from now on,? you)\b[^.\n]{0,60}"
        r"\b(dan|jailbroken|unrestricted|no restrictions|no rules|developer mode|"
        r"without (?:any )?(?:rules|restrictions|filters|limits))\b",
        _FLAGS,
    ),
    "jailbreak_keywords": re.compile(r"\b(jailbreak|dan mode|developer mode enabled)\b", _FLAGS),
    # Forging the <document> fences that wrap retrieved knowledge-base text.
    "fence_spoofing": re.compile(r"</?\s*document\b", _FLAGS),
}


def scan_prompt(text: str) -> list[str]:
    """Return the names of every injection rule the (normalised) text matches."""
    return [name for name, pattern in INJECTION_RULES.items() if pattern.search(text)]
