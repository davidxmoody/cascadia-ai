from typing import NamedTuple
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.positions import HexPosition


class Tile(NamedTuple):
    habitats: tuple[Habitat, Habitat]
    wildlife_slots: frozenset[Wildlife]
    rotation: int = 0

    @classmethod
    def from_definition(cls, definition: str, rotation: int = 0):
        h1, h2, *ws = definition
        habitats = (Habitat(h1), Habitat(h2))
        wildlife_slots = frozenset(Wildlife(w) for w in ws)
        return cls(habitats, wildlife_slots, rotation)

    @property
    def single_habitat(self):
        return self.habitats[0] == self.habitats[1]

    @property
    def unique_habitats(self):
        return set(self.habitats)

    @property
    def nature_token_reward(self):
        return self.single_habitat

    @property
    def edges(self):
        h1, h2 = self.habitats
        rot = self.rotation

        return [
            h1 if 1 <= rot <= 3 else h2,
            h1 if 2 <= rot <= 4 else h2,
            h1 if 3 <= rot <= 5 else h2,
            h2 if 1 <= rot <= 3 else h1,
            h2 if 2 <= rot <= 4 else h1,
            h2 if 3 <= rot <= 5 else h1,
        ]

    def rotate(self, steps: int):
        return self._replace(rotation=(self.rotation + steps) % 6)

    def get_edge(self, dpos: tuple[int, int]):
        h1, h2 = self.habitats
        rot = self.rotation

        match dpos:
            case (0, 1):
                return h1 if 1 <= rot <= 3 else h2
            case (1, 0):
                return h1 if 2 <= rot <= 4 else h2
            case (1, -1):
                return h1 if 3 <= rot <= 5 else h2
            case (0, -1):
                return h2 if 1 <= rot <= 3 else h1
            case (-1, 0):
                return h2 if 2 <= rot <= 4 else h1
            case (-1, 1):
                return h2 if 3 <= rot <= 5 else h1
            case _:
                raise Exception("Get edge called with non adjacent dpos")

    def __repr__(self):
        h1, h2 = (h.value for h in self.habitats)
        ws = "".join(w.value for w in self.wildlife_slots)
        return f"Tile({h1}{h2}{ws}{self.rotation})"


tile_defs = [
    "FFb",
    "FFb",
    "FFe",
    "FFf",
    "FFf",
    "FPbe",
    "FPbf",
    "FPef",
    "FPes",
    "FPfs",
    "FPsef",
    "FRbs",
    "FReb",
    "FReh",
    "FRfb",
    "FRfs",
    "FRhef",
    "FWbf",
    "FWbs",
    "FWeh",
    "FWes",
    "FWfh",
    "FWseh",
    "MFbf",
    "MFef",
    "MFfeb",
    "MFhb",
    "MFhe",
    "MFheb",
    "MMb",
    "MMe",
    "MMe",
    "MMh",
    "MMh",
    "MPbs",
    "MPes",
    "MPfeb",
    "MPhe",
    "MPhf",
    "MPsbf",
    "MWbs",
    "MWef",
    "MWeh",
    "MWfbh",
    "MWhs",
    "MWseb",
    "PPe",
    "PPe",
    "PPf",
    "PPs",
    "PPs",
    "PReh",
    "PRes",
    "PRfb",
    "PRfbh",
    "PRfh",
    "PRsbf",
    "PWef",
    "PWes",
    "PWfh",
    "PWsef",
    "PWsh",
    "PWshf",
    "RMbe",
    "RMhb",
    "RMhe",
    "RMsb",
    "RMsbh",
    "RMsh",
    "RRb",
    "RRb",
    "RRh",
    "RRh",
    "RRs",
    "WRfh",
    "WRfs",
    "WRhb",
    "WRsb",
    "WRsbh",
    "WRsh",
    "WWf",
    "WWf",
    "WWh",
    "WWs",
    "WWs",
]

tiles = [Tile.from_definition(td) for td in tile_defs]

starting_tile_defs: list[tuple[str, str, str]] = [
    ("MMb", "FWhef", "RPsb"),
    ("RRs", "PFseb", "MWfh"),
    ("PPf", "WRshf", "FMbe"),
    ("FFe", "MRheb", "WPfs"),
    ("WWh", "RFseh", "PMbf"),
]

starting_tiles: list[dict[HexPosition, Tile]] = [
    {
        (20, 21): Tile.from_definition(a, 0),
        (20, 20): Tile.from_definition(b, 1),
        (21, 20): Tile.from_definition(c, 2),
    }
    for a, b, c in starting_tile_defs
]
