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

    def __contains__(self, key: HexPosition):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def copy(self):
        return HexGrid(self._data.copy())

    def adjacent_positions(self, key: HexPosition):
        q, r = key
        for dq, dr in hex_steps:
            yield (q + dq, r + dr)

    def adjacent(self, key: HexPosition):
        for p in self.adjacent_positions(key):
            if p in self._data:
                yield (p, self._data[p])

    def all_empty_adjacent(self):
        result = set[HexPosition]()
        for p in self.keys():
            for a in self.adjacent_positions(p):
                if a not in self._data:
                    result.add(a)
        return result

    def filter(self, value: T):
        return HexGrid({p: v for p, v in self._data.items() if v == value})

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "\n".join(f"{key}: {value}" for key, value in self._data.items())
