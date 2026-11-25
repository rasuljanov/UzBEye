"""Recognize simple instance requests in Uzbek and English demo prompts."""

import re

from src.data.preprocessing import normalize

MAX_INSTANCES = 10
NUMBERS = {
    "ikki": 2, "uch": 3, "to'rt": 4, "besh": 5, "olti": 6, "yetti": 7, "sakkiz": 8,
    "to'qqiz": 9, "o'n": 10, "ikkala": 2, "uchala": 3, "both": 2,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
}
ALL = {"barcha", "hamma", "all"}
QUANTIFIER = re.compile(
    rf"^(\d{{1,3}}|(?:{'|'.join(NUMBERS)})(?:ta)?|barcha|hamma|all)\s+(?:(?:ta|the)\s+)?(.+)$"
)
RELATIONAL_MODIFIER = re.compile(
    r"\b(?:orasidagi|yonidagi|oldidagi|orqasidagi|ortidagi|ustidagi|ostidagi|tepasidagi|chapidagi|o'ngidagi)\b"
)
SINGULAR = {
    "people": "person", "persons": "person", "men": "man", "women": "woman",
    "children": "child", "cars": "car", "cats": "cat", "dogs": "dog", "chairs": "chair",
    "umbrellas": "umbrella", "bottles": "bottle", "buses": "bus", "boxes": "box",
}


def multiple_instances(prompt):
    """Return a detection phrase and optional requested count for an object set."""
    phrase = normalize(prompt).casefold().rstrip(" .!?")
    if RELATIONAL_MODIFIER.search(phrase):
        return None
    match = QUANTIFIER.fullmatch(phrase)
    count = None
    if match:
        quantity, phrase = match.groups()
        if quantity not in ALL:
            count = int(quantity) if quantity.isdecimal() else NUMBERS[quantity.removesuffix("ta")]
            if count < 2:
                return None
    words = phrase.split()
    noun = words[-1] if words else ""
    singular = SINGULAR.get(noun)
    if not singular and len(noun) > 4 and noun.endswith("lar"):
        singular = noun[:-3]
    if not match and (len(words) != 1 or not singular):
        return None
    if singular:
        words[-1] = singular
    return " ".join(words), count
