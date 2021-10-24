# cah-ai

This is a Cards Against Humanity AI implemented using a pre-trained Semantic Search model.

# How it works

A player is described by a combination of a text description (e.g. "a college frat boy") and a `personality_power` (which essentially controls how much their personality description affects their actions). The players are implemented using a semantic similarity model. Prompts are encoded on a per-player basis (e.g. as "a college frat boy answered question ... as") and answers are encoded as normal. The player's choice is determined probabilistically using the semantic similarity model.

# Results

I tried simulating a game using [scripts/simulate_game.py](scripts/simulate_game.py). I had four virtual players: three with text descriptions, and one that takes random actions. Here are the players and their resulting number of wins, after going through the whole deck:

 * 395 wins - `DescPlayer("a college frat boy", personality_power=5.0)`
 * 180 wins - `DescPlayer("a middle-aged man", personality_power=5.0)`
 * 169 wins - `DescPlayer("an old racist southern lady", personality_power=5.0)`
 * 213 wins - `RandomPlayer()`

Interestingly, the fictional college frat boy won by a large margin, whereas the other three players did considerably worse. More interestingly, the random player seems to be better than the two losing personality-based players. I have tried tweaking personality powers for the two losing players, and decreasing their personality power slightly helps them, but not enough to definitively beat the random player.
