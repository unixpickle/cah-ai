# cah-ai

This is a Cards Against Humanity AI implemented using a pre-trained Semantic Search model.

# How it works

A player is described by a combination of a text description (e.g. "a college frat boy") and a `personality_power` (which essentially controls how much their personality description affects their actions). The players are implemented using a semantic similarity model. Prompts are encoded on a per-player basis (e.g. as "a college frat boy answered question ... as") and answers are encoded as normal. The player's choice is determined probabilistically using the semantic similarity model.

# Results

I tried simulating a game using [scripts/simulate_game.py](scripts/simulate_game.py). I had four virtual players: three with text descriptions, and one that takes random actions. Here are the players and their resulting number of wins, after going through the whole deck:

```
278 wins - DescPlayer("a college frat boy", personality_power=5.0)
232 wins - DescPlayer("a middle-aged man", personality_power=5.0)
236 wins - DescPlayer("an old racist southern lady", personality_power=5.0)
211 wins - RandomPlayer()
```

Interestingly, the random player is only a tiny bit worse, even though the other three players are implemented the same way and are therefore more likely to select winners amongst themselves.

Let's see what happens in a game with three duplicate AI players and one random player:

```
267 wins - DescPlayer("a college frat boy", personality_power=5.0)
247 wins - DescPlayer("a college frat boy", personality_power=5.0)
249 wins - DescPlayer("a college frat boy", personality_power=5.0)
194 wins - RandomPlayer()
```

Surprisingly, the random player's wins didn't decrease very much. Maybe it's because the players are stochastic, Let's try reducing their temperature to `0.1`:

```
270 wins - DescPlayer("a college frat boy", personality_power=5.0, temperature=0.1)
251 wins - DescPlayer("a college frat boy", personality_power=5.0, temperature=0.1)
254 wins - DescPlayer("a college frat boy", personality_power=5.0, temperature=0.1)
182 wins - RandomPlayer()
```

So interestingly, even if the first three players are highly "opinionated" and all agree exactly, a random player still gets 19% of the wins (25% would be the case if all players are random).
