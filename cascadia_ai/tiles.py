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

    @property
    def single_habitat(self):
        return self.habitats[0] == self.habitats[1]

    @property
    def nature_token_reward(self):
        return self.single_habitat

    def __repr__(self):
        h1, h2 = (h.value for h in self.habitats)
        ws = "".join(w.value for w in self.wildlife_slots)
        return f'Tile("{h1}{h2}{ws}")'


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
