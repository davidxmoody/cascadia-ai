import pickle
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.game_state import GameState
from cascadia_ai.habitat_layer import HabitatLayer
from cascadia_ai.score import Score
from cascadia_ai.wildlife_layer import FoxLayer, WildlifeLayer


# %%
def migrate_state(old_s):
    new_s = GameState()

    new_s._tile_supply = old_s._tile_supply
    new_s._wildlife_supply = old_s._wildlife_supply
    new_s.tile_display = old_s.tile_display
    new_s.wildlife_display = old_s.wildlife_display

    new_s.env.nature_tokens = getattr(old_s, "nature_tokens", 0)
    new_s.env.tiles = old_s.env.tiles
    new_s.env.wildlife = old_s.env.wildlife
    new_s.env.hlayers = {h: HabitatLayer(h, old_s.env.tiles) for h in Habitat}
    new_s.env.wlayers = {
        w: (
            FoxLayer(old_s.env.wildlife)
            if w == Wildlife.FOX
            else WildlifeLayer(w, old_s.env.wildlife)
        )
        for w in Wildlife
    }
    return new_s


# %%
with open("data/old_realistic_states.pkl", "rb") as f:
    old_realistic_states = pickle.load(f)


# %%
new_realistic_states = [migrate_state(s) for s in old_realistic_states]

with open("data/realistic_states.pkl", "wb") as f:
    pickle.dump(new_realistic_states, f)


# %%
with open("data/old_greedy_played_games.pkl", "rb") as f:
    old_greedy = pickle.load(f)


# %%
new_greedy = []
for state, scores in old_greedy:
    new_state = migrate_state(state)
    new_scores = [(s.wildlife, s.habitat, s.nature_tokens) for s in scores]
    new_greedy.append((new_state, new_scores))


# %%
new_fixed_score_greedy = []
for state, scores in new_greedy:
    new_fixed_score_greedy.append((state, [Score(*s) for s in scores]))


# %%
with open("data/greedy_played_games.pkl", "wb") as f:
    pickle.dump(new_fixed_score_greedy, f)
