from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import Environment, RotatedTile
from cascadia_ai.hex_grid import HexGrid
from cascadia_ai.score import calculate_habitat_score, calculate_bears_score
from cascadia_ai.tiles import Tile


def make_env(
    tiles: list[tuple[tuple[int, int], str, int]],
    wildlife: list[tuple[tuple[int, int], Wildlife]] = [],
):
    env = Environment(
        HexGrid(
            {p: RotatedTile(Tile.from_definition(tdef), rot) for p, tdef, rot in tiles}
        )
    )

    for p, w in wildlife:
        env.place_wildlife(p, w)

    return env


def test_habitat_score():
    env = make_env(
        [
            ((0, 0), "MMb", 0),
            ((1, -1), "FWhef", 1),
            ((0, -1), "RPsb", 2),
            ((1, 0), "MRf", 1),
            ((-1, 0), "RRf", 0),
        ]
    )

    score = calculate_habitat_score(env.tiles)
    assert score[Habitat("M")] == 2
    assert score[Habitat("F")] == 1
    assert score[Habitat("P")] == 1
    assert score[Habitat("W")] == 1
    assert score[Habitat("R")] == 2


def test_habitat_score_bonus():
    area_size = 10
    env = make_env([((i, 0), "MMb", 0) for i in range(area_size)])

    score = calculate_habitat_score(env.tiles)
    assert score[Habitat("M")] == area_size + 2


def test_bears_score():
    env = make_env([((q, r), "MMb", 0) for q in range(3) for r in range(11)])
    assert calculate_bears_score(env.wildlife) == 0

    env.place_wildlife((0, 0), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 0

    env.place_wildlife((1, 0), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 4

    env.place_wildlife((2, 0), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 0

    env.place_wildlife((0, 2), Wildlife.BEAR)
    env.place_wildlife((1, 2), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 4

    env.place_wildlife((0, 4), Wildlife.BEAR)
    env.place_wildlife((1, 4), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 11

    env.place_wildlife((0, 6), Wildlife.BEAR)
    env.place_wildlife((1, 6), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 19

    env.place_wildlife((0, 8), Wildlife.BEAR)
    env.place_wildlife((1, 8), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 27

    env.place_wildlife((0, 10), Wildlife.BEAR)
    env.place_wildlife((1, 10), Wildlife.BEAR)
    assert calculate_bears_score(env.wildlife) == 27
