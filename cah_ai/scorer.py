from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .player import Player


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
    ) -> np.ndarray:
        """
        For each player, predict the probability of each answer.

        Player index is the row, and answer is the column: x[player, answer].
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
        return np.stack(results, axis=0)
