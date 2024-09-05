from typing import NamedTuple
from cascadia_ai.enums import Habitat, Wildlife


class Tile(NamedTuple):
    habitats: tuple[Habitat, Habitat]
    wildlife_slots: frozenset[Wildlife]

    @classmethod
    def from_definition(cls, definition: str):
        h1, h2, *ws = definition
        habitats = (Habitat(h1), Habitat(h2))
        wildlife_slots = frozenset(Wildlife(w) for w in ws)
        return cls(habitats, wildlife_slots)

    def nature_token_reward(self):
        return self.habitats[0] == self.habitats[1]


tile_defs = [
    # TODO add all tiles
    "RRb",
    "MWeh",
]

tiles = [Tile.from_definition(td) for td in tile_defs]
