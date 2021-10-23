import gzip
import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Prompt:
    text: str
    pick: int


@dataclass
class Deck:
    prompts: List[Prompt]
    answers: List[str]


def load_cards(path: Optional[str] = None) -> Deck:
    """
    Load a collection of prompts and answers from a .json.gz file.

    If no path is specified, this uses the set of official cards from
    https://crhallberg.com/cah/.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "cah-cards-compact.json.gz")
    with gzip.open(path, "rb") as f:
        data = json.loads(f.read())
    return Deck(prompts=[Prompt(**x) for x in data["black"]], answers=data["white"])
