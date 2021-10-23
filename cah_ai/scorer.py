from abc import ABC, abstractmethod

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer, util


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
    ) -> List[float]:
        """
        Given the encodings for a prompt and the encoded answers, compute a
        list of probabilities for each answer.
        """


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
    ) -> List[float]:
        dots = {
            k: np.array([v @ answer["generic"] for answer in answers])
            for k, v in prompt.items()
        }
        scores = dots["generic"] + self.personality_power * (
            dots["desc"] - dots["generic"]
        )
        return (scores / np.sum(scores)).tolist()


class Scorer:
    """
    A wrapper around a semantic search model that computes feature vectors for
    Players to make decisions.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def scores(
        self,
        prompt: str,
        players: List[Player],
        answers: List[str],
    ) -> List[List[float]]:
        """
        For each player, predict the probability of each answer.
        """
        prompt_strs = set()
        enc_prompts = []
        enc_answers = []
        for player in players:
            enc_prompts.append(player.encode_prompt(prompt))
            enc_answers.append([player.encode_answer(a) for a in answers])
        for p in enc_prompts:
            prompt_strs.update(p.values())
        for xs in enc_answers:
            for x in xs:
                prompt_strs.update(x.values())

        ordered_prompts = list(prompt_strs)
        embs = self.model.encode(ordered_prompts)
        text_to_emb = dict(zip(ordered_prompts, embs))

        results = []
        for player, player_prompt, player_answers in zip(
            players, enc_prompts, enc_answers
        ):
            prompt_vecs = {k: text_to_emb[v] for k, v in player_prompt.items()}
            answers_vecs = [
                {k: text_to_emb[v] for k, v in x.items()} for x in player_answers
            ]
            results.append(player.score_answers(prompt_vecs, answers_vecs))
        return results
