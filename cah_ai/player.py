import random
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Player(ABC):
    """
    A Player chooses answers to prompt cards based on a "personality".

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

    @abstractmethod
    def choose_answer(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> int:
        """
        Like score_answers(), but select an answer as a judge.
        """


class LexPlayer(Player):
    """
    A Player that uses lexicographical ordering.
    """

    def encode_prompt(self, prompt: str) -> Dict[str, str]:
        return {prompt: ""}

    def encode_answer(self, answer: str) -> Dict[str, str]:
        return {answer: ""}

    def score_answers(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        items = sorted(enumerate(answers), key=lambda x: list(x[1].keys())[0])
        res = np.zeros([len(answers)])
        res[items[0][0]] = 1
        return res

    def choose_answer(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> int:
        items = sorted(enumerate(answers), key=lambda x: list(x[1].keys())[0])
        return items[0][0]


class RandomPlayer(Player):
    def encode_prompt(self, prompt: str) -> Dict[str, str]:
        return {}

    def encode_answer(self, answer: str) -> Dict[str, str]:
        return {}

    def score_answers(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        return np.ones(len(answers)) / len(answers)

    def choose_answer(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> int:
        return random.randrange(len(answers))


class DescPlayer(Player):
    """
    A Player whose behavior is determined by a prefix string.

    The weight of the prefix string in the decision process is determined by
    personality_power. With a value of 1.0, the string is merely used as a
    prefix for a normal semantic search. With a value greater than 1.0, the
    effect of the prefix string is amplified, and the reverse is true for
    values less than 1.0.
    """

    def __init__(
        self,
        description: str,
        personality_power: float = 1.0,
        temperature: float = 1.0,
        guide_cah: bool = False,
    ):
        self.description = description
        self.personality_power = personality_power
        self.temperature = temperature
        self.guide_cah = guide_cah

    def encode_prompt(self, prompt: str) -> str:
        return {
            "desc": f'{self.description} answer to Cards Against Humanity prompt, "{prompt}".',
            "generic": prompt
            if self.guide_cah
            else f'answer to Cards Against Humanity prompt, "{prompt}".',
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
        scores /= max(1e-5, self.temperature)
        scores = np.exp(scores - np.max(scores))
        return scores / np.sum(scores)

    def choose_answer(
        self, prompt: Dict[str, np.ndarray], answers: List[Dict[str, np.ndarray]]
    ) -> int:
        scores = self.score_answers(prompt, answers)
        return np.argmax(scores)
