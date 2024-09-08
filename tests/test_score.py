from cascadia_ai.enums import Habitat
from cascadia_ai.environments import Environment, RotatedTile
from cascadia_ai.hex_grid import HexGrid
from cascadia_ai.score import calculate_habitat_score
from cascadia_ai.tiles import Tile


def make_env(placements: list[tuple[tuple[int, int], str, int]]):
    return Environment(
        HexGrid(
            {
                p: RotatedTile(Tile.from_definition(tdef), rot)
                for p, tdef, rot in placements
            }
        )
    )


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
