HexPosition = tuple[int, int]

adjacent_positions_cache: dict[HexPosition, tuple[HexPosition, ...]] = {}


def adjacent_positions(pos: HexPosition):
    if pos not in adjacent_positions_cache:
        q, r = pos
        adjacent_positions_cache[pos] = (
            (q, r + 1),
            (q + 1, r),
            (q + 1, r - 1),
            (q, r - 1),
            (q - 1, r),
            (q - 1, r + 1),
        )
    return adjacent_positions_cache[pos]
