from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Player(ABC):
    """
    A Player choses answers to prompt cards based on a "personality".

    The encode_prompt() and encode_answer() methods request feature vectors for
    each prompt and answer card. For example, the Player might make decisions
    using features for a prompt with and without some prefix added on.

    The score_answers() method decides the probability distribution over
    answers, given embedding vectors for all of the strings requested by the
    encode_*() methods.
    """

    @abstractmethod
    def encode_prompt(self, prompt: str) -> Dict[str, str]:
        """
        Create one or more encodings of the prompt, where each
        encoding has a name (a key) and a corresponding sentence.
        """

    @abstractmethod
    def encode_answer(self, answer: str) -> Dict[str, str]:
        """
        Create one or more encodings of the answer. Same return value structure
        as encode_prompt().
        """

    @abstractmethod
    def score_answers(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Given the encodings for a prompt and the encoded answers, compute an
        array of probabilities (one for each answer).
        """


class RandomPlayer(Player):
    def encode_prompt(self, prompt: str) -> Dict[str, str]:
        return {}

    def encode_answer(self, answer: str) -> Dict[str, str]:
        return {}

    def score_answers(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        return np.ones(len(answers)) / len(answers)


class DescPlayer(Player):
    """
    A Player whose behavior is determined by a prefix string.

    The weight of the prefix string in the decision process is determined by
    personality_power. With a value of 1.0, the string is merely used as a
    prefix for a normal semantic search. With a value greater than 1.0, the
    effect of the prefix string is amplified, and the reverse is true for
    values less than 1.0.
    """

    def __init__(self, description: str, personality_power: float = 1.0):
        self.description = description
        self.personality_power = personality_power

    def encode_prompt(self, prompt: str) -> str:
        return {
            "desc": f'{self.description} answer to Cards Against Humanity prompt, "{prompt}".',
            "generic": prompt,
        }

    def encode_answer(self, answer: str) -> str:
        return {
            "generic": answer,
        }

    def score_answers(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        logs = {
            k: np.log(
                np.maximum(
                    1e-5, np.array([v @ answer["generic"] for answer in answers])
                )
            )
            for k, v in prompt.items()
        }
        scores = logs["generic"] + self.personality_power * (
            logs["desc"] - logs["generic"]
        )
        scores = np.exp(scores)
        return scores / np.sum(scores)
