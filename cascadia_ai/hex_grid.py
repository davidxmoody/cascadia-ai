from typing import Generic, TypeVar


type HexPosition = tuple[int, int]

hex_steps: list[tuple[int, int]] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


T = TypeVar("T")


class HexGrid(Generic[T]):
    _data: dict[HexPosition, T]

    def __init__(self, data: None | dict[HexPosition, T] = None):
        self._data = data or {}

    def __getitem__(self, key: HexPosition):
        return self._data[key] if key in self._data else None

    def __setitem__(self, key: HexPosition, value: T):
        if key in self._data:
            raise KeyError("Cannot overwrite existing data")
        self._data[key] = value

    def items(self):
        return self._data.items()

    def copy(self):
        return HexGrid(self._data.copy())

    def adjacent(self, key: HexPosition):
        q, r = key
        for dq, dr in hex_steps:
            apos = (q + dq, r + dr)
            if apos in self._data:
                yield (apos, self._data[apos])

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "\n".join(f"{key}: {value}" for key, value in self._data.items())
