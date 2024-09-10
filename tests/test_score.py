from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import Environment, RotatedTile
from cascadia_ai.hex_grid import HexGrid
from cascadia_ai.score import (
    calculate_habitat_score,
    calculate_bear_score,
    calculate_elk_score,
    calculate_salmon_score,
    calculate_hawk_score,
    calculate_fox_score,
)
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

    assert calculate_habitat_score(Habitat("M"), env.tiles) == 2
    assert calculate_habitat_score(Habitat("F"), env.tiles) == 1
    assert calculate_habitat_score(Habitat("P"), env.tiles) == 1
    assert calculate_habitat_score(Habitat("W"), env.tiles) == 1
    assert calculate_habitat_score(Habitat("R"), env.tiles) == 2


def test_habitat_score_bonus():
    area_size = 10
    env = make_env([((i, 0), "MMb", 0) for i in range(area_size)])

    assert calculate_habitat_score(Habitat("M"), env.tiles) == area_size + 2


def test_bear_score():
    env = make_env([((q, r), "MMb", 0) for q in range(3) for r in range(11)])
    assert calculate_bear_score(env.wildlife) == 0

    env.place_wildlife((0, 0), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 0

    env.place_wildlife((1, 0), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 4

    env.place_wildlife((2, 0), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 0

    env.place_wildlife((0, 2), Wildlife.BEAR)
    env.place_wildlife((1, 2), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 4

    env.place_wildlife((0, 4), Wildlife.BEAR)
    env.place_wildlife((1, 4), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 11

    env.place_wildlife((0, 6), Wildlife.BEAR)
    env.place_wildlife((1, 6), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 19

    env.place_wildlife((0, 8), Wildlife.BEAR)
    env.place_wildlife((1, 8), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 27

    env.place_wildlife((0, 10), Wildlife.BEAR)
    env.place_wildlife((1, 10), Wildlife.BEAR)
    assert calculate_bear_score(env.wildlife) == 27


def test_elk_score():
    env = make_env([((q, r), "MMe", 0) for q in range(10) for r in range(10)])
    assert calculate_elk_score(env.wildlife) == 0

    env.place_wildlife((0, 0), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 2

    env.place_wildlife((1, 0), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 5

    env.place_wildlife((2, 0), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 9

    env.place_wildlife((3, 0), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 13

    env.place_wildlife((4, 0), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 13 + 2

    env.place_wildlife((4, 1), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 13 + 5

    env.place_wildlife((6, 6), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 13 + 5 + 2

    env.place_wildlife((0, 1), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 13 + 5 + 2 + 2

    env.place_wildlife((0, 2), Wildlife.ELK)
    assert calculate_elk_score(env.wildlife) == 13 + 9 + 2 + 2


def test_salmon_score():
    env = make_env([((q, r), "MMs", 0) for q in range(10) for r in range(10)])
    assert calculate_salmon_score(env.wildlife) == 0

    env.place_wildlife((0, 0), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 2

    env.place_wildlife((1, 0), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 5

    env.place_wildlife((1, 1), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 8

    env.place_wildlife((2, 0), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 0

    env.place_wildlife((4, 0), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 2

    env.place_wildlife((6, 0), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 2 + 2

    for i in range(1, 7):
        env.place_wildlife((6, i), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 2 + 25

    env.place_wildlife((6, 7), Wildlife.SALMON)
    assert calculate_salmon_score(env.wildlife) == 2 + 25


def test_hawk_score():
    env = make_env([((q, r), "MMh", 0) for q in range(10) for r in range(10)])
    assert calculate_hawk_score(env.wildlife) == 0

    env.place_wildlife((0, 0), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 2

    env.place_wildlife((1, 0), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 0

    env.place_wildlife((0, 2), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 2

    env.place_wildlife((2, 2), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 5

    env.place_wildlife((2, 4), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 8

    env.place_wildlife((4, 4), Wildlife.HAWK)
    env.place_wildlife((4, 6), Wildlife.HAWK)
    env.place_wildlife((6, 6), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 18

    env.place_wildlife((6, 8), Wildlife.HAWK)
    env.place_wildlife((8, 8), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 26

    env.place_wildlife((0, 8), Wildlife.HAWK)
    assert calculate_hawk_score(env.wildlife) == 26


def test_fox_scoring():
    env = make_env([((q, r), "MMbeshf", 0) for q in range(10) for r in range(10)])
    assert calculate_fox_score(env.wildlife) == 0

    env.place_wildlife((5, 5), Wildlife.FOX)
    assert calculate_fox_score(env.wildlife) == 0

    env.place_wildlife((6, 5), Wildlife.HAWK)
    assert calculate_fox_score(env.wildlife) == 1

    env.place_wildlife((5, 6), Wildlife.HAWK)
    assert calculate_fox_score(env.wildlife) == 1

    env.place_wildlife((4, 6), Wildlife.BEAR)
    assert calculate_fox_score(env.wildlife) == 2

    env.place_wildlife((4, 5), Wildlife.SALMON)
    assert calculate_fox_score(env.wildlife) == 3

    env.place_wildlife((5, 4), Wildlife.ELK)
    assert calculate_fox_score(env.wildlife) == 4

    env.place_wildlife((6, 4), Wildlife.FOX)
    assert calculate_fox_score(env.wildlife) == 5 + 3
