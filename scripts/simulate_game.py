"""
Simulate many rounds of games to see which personality would win in a
long-played game of CAH.
"""

import itertools
import random
from typing import List, Tuple

import numpy as np
from cah_ai import Deck, DescPlayer, RandomPlayer, Scorer


def main():
    players = [
        DescPlayer("a college frat boy", personality_power=5.0),
        DescPlayer("a middle-aged man", personality_power=5.0),
        DescPlayer("an old racist southern lady", personality_power=5.0),
        RandomPlayer(),
    ]

    print("Setting up deck...")
    deck = Deck.load()
    random.shuffle(deck.answers)
    random.shuffle(deck.prompts)

    print("Dealing...")
    player_hands = []
    for _ in players:
        player_hands.append(deck.answers[:7])
        deck.answers = deck.answers[7:]

    print("Creating scorer model...")
    scorer = Scorer()

    print("Simulating game...")
    tally = [0] * len(players)
    judge = 0
    while len(deck.prompts):
        prompt = deck.prompts.pop()
        played_answers = []
        played_players = []
        for i, (player, hand) in enumerate(zip(players, player_hands)):
            if i == judge:
                continue

            # Brute-force pick the best combination of answer cards.
            # Usually the answer is just one card, but not always.
            combos, indices = answer_combinations(hand, prompt.pick)
            probs = scorer.scores(prompt.text, [player], combos)[0]
            choice = np.random.choice(len(probs), p=probs)
            played_answers.append(combos[choice])
            played_players.append(i)

            for i in sorted(indices[choice], reverse=True):
                deck.answers.append(hand[i])
                del hand[i]
            while len(hand) < 7:
                hand.append(deck.answers.pop(0))

        print("-----")
        print(prompt)
        print(played_answers)

        judge_player = players[judge]
        best = np.argmax(scorer.scores(prompt.text, [judge_player], played_answers)[0])
        if best >= judge:
            best += 1
        print(f"judge={judge} best={best}")
        tally[best] += 1
        judge = (judge + 1) % len(players)

        print("win tally:", tally)


def answer_combinations(
    hand: List[str], pick: int
) -> Tuple[List[str], List[List[int]]]:
    strs = []
    indices = []
    for combo in itertools.product(*([range(len(hand))] * pick)):
        if len(set(combo)) != len(combo):
            # Using one card multiple times.
            continue
        strs.append(" ".join(hand[i] for i in combo))
        indices.append(combo)
    return strs, indices


if __name__ == "__main__":
    main()
