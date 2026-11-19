"""Text fixes shared by preprocess.py and build_dataset.py.

1. Tensor Trust "access granted" masking.
   In the game, the defender's LLM must say "access granted" to open the account, so half of the
   Tensor Trust attacks contain that phrase literally ("say 'access granted'"). No other source does,
   so a detector would learn the game phrase instead of what an attack is. Deleting the phrase leaves
   empty quotes ('say ""') in ~26% of Tensor Trust attacks, a new marker of the same kind.
   We therefore REPLACE it with one of many ordinary phrases, chosen deterministically per attack,
   so sentences stay natural and no single replacement marks attacks.

2. CJK language detection.
   Short Chinese/Japanese/Korean texts are too short for the statistical language detector, which
   returns "undetermined". We decide them by script instead.
"""
from __future__ import annotations

import hashlib
import re

ACCESS_GRANTED = re.compile(r"access[\s_\-]*granted", re.IGNORECASE)

# Ordinary target phrases (no "access", no "granted"). Chosen on Nov 19 from 40 candidates by measuring
# how often each already occurs in attacks vs benign text: words typical of attack text ("Correct",
# "Hello", "Next", "OK", "Continue", "Proceed") were excluded so the replacements add no marker.
REPLACEMENTS = [
    "Welcome", "Approved", "Hello world", "Confirmed", "Yes", "Done", "Success", "I have been PWNED",
    "All clear", "Welcome back", "Authorized", "Go ahead", "Accepted", "Verified", "Sure thing",
    "Unlocked", "Entry allowed", "Permission given", "Thank you", "Great", "Good morning", "Hi there",
    "Of course", "Certainly", "Perfect", "Excellent", "Understood", "Noted", "Ready", "Very well",
    "Alright", "Got it", "Sounds good", "Fine",
]


def _match_case(template: str, phrase: str) -> str:
    if template.isupper():
        return phrase.upper()
    if template.islower():
        return phrase.lower()
    return phrase


def mask_access_granted(text: str, key: str) -> str:
    """Replace every 'access granted' with one ordinary phrase, chosen by a stable hash of `key`
    (e.g. the row id), keeping the original upper/lower case style."""
    if not ACCESS_GRANTED.search(text):
        return text
    idx = int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % len(REPLACEMENTS)
    phrase = REPLACEMENTS[idx]
    return ACCESS_GRANTED.sub(lambda m: _match_case(m.group(0), phrase), text).strip()


_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")
_HAN = re.compile(r"[㐀-鿿]")


def cjk_language(text: str, min_share: float = 0.3) -> str | None:
    """'ja' / 'ko' / 'zh' if CJK characters make up at least `min_share` of the letters, else None."""
    letters = sum(ch.isalpha() for ch in text)
    if not letters:
        return None
    kana, hangul, han = len(_KANA.findall(text)), len(_HANGUL.findall(text)), len(_HAN.findall(text))
    if (kana + hangul + han) / letters < min_share:
        return None
    if kana:
        return "ja"
    if hangul >= han:
        return "ko"
    return "zh"


# Patterns that should NOT be much more common in attacks than in benign text (checked in the dataset card)
MARKERS = {
    "empty quotes": re.compile(r'""|\'\'|“”|‘’|「」'),
    "access granted": ACCESS_GRANTED,
    **{f"'{p}'": re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in REPLACEMENTS},
}
