"""Conservative, auditable rule suggestions for APIG strategy labelling.

This module does NOT decide ground truth, nor does it decide NOT_ATTACK from an
absence of matched rules. Source-labelled attacks can contain harmless rows,
and unrecognised attacks may use languages or mechanisms not covered here.
All outputs are *suggestions* pending review.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Pattern


def _patterns(expressions: tuple[str, ...]) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(x, re.IGNORECASE | re.MULTILINE) for x in expressions)


# Narrow cues: language-independent coverage is NOT claimed.
S1 = _patterns((
    r"\b(?:ignore|disregard|forget|discard|override)\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|earlier|above|system|developer|original|your)\s+(?:instructions?|rules?|prompts?|tasks?|directives?|constraints?|guidelines?|information|assignments?)\b",
    r"\b(?:forget|ignore|disregard)\s+(?:everything|all)\s+(?:you\s+were\s+told|above|before)\b",
    r"\b(?:do\s+not|don't|stop)\s+(?:follow|following|obey|obeying)\s+(?:the\s+)?(?:previous|system|developer|original)\s+(?:instructions?|rules?|prompts?)\b",
    r"\b(?:reveal|print|show|expose|leak)\s+(?:me\s+)?(?:all\s+)?(?:your\s+)?(?:hidden|secret|system|developer|internal|original)\s+(?:prompt|instructions?|password|rules?)\b",
))
S2 = _patterns((
    r"\b(?:you\s+are\s+(?:now|no\s+longer)|act\s+as|pretend\s+(?:to\s+be|you\s+are)|role\s*play\s+as|assume\s+the\s+role\s+of)\b.{0,200}\b(?:unrestricted|unfiltered|uncensored|without\s+(?:any\s+)?(?:restrictions|rules|limitations)|ignore\s+(?:safety|policy)|bypass\s+(?:safety|restrictions))\b",
    r"\b(?:DAN|do\s+anything\s+now|developer\s+mode)\b.{0,180}\b(?:no\s+(?:rules|restrictions)|unrestricted|policy|safety|filter|bypass)\b",
    r"\b(?:simulation|fictional|roleplay|role-play)\b.{0,180}\b(?:bypass|no\s+(?:rules|restrictions)|unrestricted|ignore\s+(?:safety|policy))\b",
))
S3 = _patterns((
    r"(?:\[/?(?:system|developer|assistant)\]|<\|(?:im_start|im_end)\|>|\[INST\]|<<SYS>>|\[start_header_id\])",
    r"(?:^|\n)\s*(?:system|developer)\s*(?:message|instruction|prompt)\s*:\s*\S+",
    r"\b(?:the\s+(?:real|actual)\s+system\s+(?:message|instruction)|new\s+system\s+(?:message|instruction))\s*(?::|is|says)\b",
))
OBFUSCATION = _patterns((
    r"\b(?:base64|rot13|hex(?:adecimal)?|unicode\s+escape)\b.{0,100}\b(?:decode|encoded|decrypt|interpret|execute|follow)\b",
    r"\b(?:decode|decrypt)\b.{0,100}\b(?:base64|rot13|hex(?:adecimal)?)\b",
))


@dataclass(frozen=True)
class Suggestion:
    strategy_primary: str | None
    strategy_secondary: tuple[str, ...]
    ambiguous: bool
    obfuscated: bool
    status: str                 # 'rule_candidate' or 'needs_review'
    matched_rules: tuple[str, ...]
    notes: str


def suggest(text: str) -> Suggestion:
    """Find high-specificity strategy cues, returning a reviewable *candidate*.

    Multiple strategy families are not automatically resolved to a primary:
    the most important mechanism requires contextual interpretation.
    """
    if not isinstance(text, str) or not text.strip():
        return Suggestion(None, (), False, False, 'needs_review', (), 'Empty or invalid text')
    matches = {
        'S1': any(p.search(text) for p in S1),
        'S2': any(p.search(text) for p in S2),
        'S3': any(p.search(text) for p in S3),
    }
    hits = tuple(k for k in ('S1', 'S2', 'S3') if matches[k])
    obfuscated = any(p.search(text) for p in OBFUSCATION)
    if len(hits) == 1:
        return Suggestion(hits[0], (), False, obfuscated, 'rule_candidate', hits,
                          'Single rule-family match; verify attack validity and strategy')
    if len(hits) >= 2:
        return Suggestion(None, hits, True, obfuscated, 'needs_review', hits,
                          'Multiple strategy cues: human/LLM must choose primary and secondary')
    return Suggestion(None, (), False, obfuscated, 'needs_review', (),
                      'No high-specificity pattern: may be attack, NOT_ATTACK, or multilingual')
