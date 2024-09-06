import unittest
from cascadia_ai.enums import Habitat
from cascadia_ai.environments import Environment, starting_tiles
from cascadia_ai.score import calculate_habitat_score
from cascadia_ai.tiles import Tile


class TestScore(unittest.TestCase):
    def test_habitat_score(self):
        env = Environment(starting_tiles[0])
        env.place_tile((1, 0), Tile.from_definition("MRf"), 1)
        env.place_tile((-1, 0), Tile.from_definition("RRf"), 0)
        env.print()

        score = calculate_habitat_score(env.tiles)

        self.assertEqual(
            score,
            {
                Habitat("M"): 2,
                Habitat("F"): 1,
                Habitat("P"): 1,
                Habitat("W"): 1,
                Habitat("R"): 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
