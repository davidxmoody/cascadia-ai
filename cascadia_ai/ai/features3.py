from collections import Counter, defaultdict
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from cascadia_ai.ai.actions import (
    calculate_bear_reward,
    calculate_elk_reward,
    calculate_hawk_reward,
    calculate_salmon_reward,
)
from cascadia_ai.enums import Habitat, Wildlife
from cascadia_ai.environments import (
    HexPosition,
    adjacent_positions,
    is_adjacent,
    share_edge,
)
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.tiles import Tile

feature_names = [
    "turns_remaining",
    "nature_tokens",
    # "unclaimed_nt_rewards",
    # *(f"{w.value}_num_unoccupied_slots" for w in Wildlife),
    "num_bear_pairs",
    "num_bear_singles",
    "num_elk",
    "num_salmon",
    "num_hawks",
    "num_foxes",
    # "usable_spaces",
    # *(f"{w.value}_best_adjacent_wreward" for w in Wildlife),
    # *(f"{w.value}_best_unoccupied_wreward" for w in Wildlife),
    # *(f"{h.value}_best_hreward" for h in Habitat),
    *(f"{h.value}_largest_area" for h in Habitat),
]

F = {name: i for i, name in enumerate(feature_names)}


def cached(method):
    @wraps(method)
    def wrapper(self, *args):
        if not hasattr(self, "_cache"):
            self._cache = defaultdict(dict)

        subcache = self._cache[method.__name__]

        if args in subcache:
            return subcache[args]

        result = method(self, *args)
        subcache[args] = result
        return result

    return wrapper


class StateFeatures:
    def __init__(self, state: GameState):
        self.state = state

    def _wildlife_target(self, action: Action):
        if action.wildlife_position is None:
            return None
        if action.wildlife_position == action.tile_position:
            return self.state.tile_display[action.tile_index]
        return self.state.env.tiles[action.wildlife_position]

    def _nature_tokens(self, action: Action | None):
        nt = self.state.nature_tokens
        if action is not None:
            if action.nt_spent:
                nt -= 1
            wildlife_target = self._wildlife_target(action)
            if wildlife_target is not None and wildlife_target.nature_token_reward:
                nt += 1
        return nt

    @cached
    def _wildlife_count(
        self, wildlife: Wildlife, placed_wildlife: tuple[HexPosition, Wildlife] | None
    ):
        if placed_wildlife is None or placed_wildlife[1] != wildlife:
            return sum(w == wildlife for w in self.state.env.wildlife.values())

        return self._wildlife_count(wildlife, None) + 1

    @cached
    def _wildlife_groups(
        self, wildlife: Wildlife, placed_wildlife: tuple[HexPosition, Wildlife] | None
    ):
        if placed_wildlife is None or placed_wildlife[1] != wildlife:
            return self.state.env.wildlife_groups(wildlife)

        connected_positions = set(
            pos for pos, _ in self.state.env.adjacent_wildlife(*placed_wildlife)
        )
        wildlife_groups = [{placed_wildlife[0]}]

        for group in self._wildlife_groups(wildlife, None):
            if group.isdisjoint(connected_positions):
                wildlife_groups.append(group)
            else:
                wildlife_groups[0].update(group)

        return sorted(wildlife_groups, key=len, reverse=True)

    @cached
    def _base_wreward(
        self,
        pos: HexPosition,
        wildlife: Wildlife,
        placed_wildlife: tuple[HexPosition, Wildlife] | None,
    ):
        match wildlife:
            case Wildlife.BEAR:
                groups = self._wildlife_groups(Wildlife.BEAR, placed_wildlife)
                return calculate_bear_reward(groups, pos)

            case Wildlife.ELK:
                groups = self._wildlife_groups(Wildlife.ELK, placed_wildlife)
                return calculate_elk_reward(groups, pos)

            case Wildlife.SALMON:
                groups = self._wildlife_groups(Wildlife.SALMON, placed_wildlife)
                return calculate_salmon_reward(groups, pos)

            case Wildlife.HAWK:
                groups = self._wildlife_groups(Wildlife.HAWK, placed_wildlife)
                return calculate_hawk_reward(groups, pos)

            case Wildlife.FOX:
                adjacent_wildlife = {
                    w for _, w in self.state.env.adjacent_wildlife(pos)
                }
                if placed_wildlife is not None and is_adjacent(placed_wildlife[0], pos):
                    adjacent_wildlife.add(placed_wildlife[1])
                return len(adjacent_wildlife)

    @cached
    def _adjacent_fox_wreward(
        self,
        pos: HexPosition,
        wildlife: Wildlife,
        placed_wildlife: tuple[HexPosition, Wildlife] | None,
    ):
        adjacent_fox_positions = {
            fp for fp, _ in self.state.env.adjacent_wildlife(pos, Wildlife.FOX)
        }
        if (
            placed_wildlife is not None
            and placed_wildlife[1] == Wildlife.FOX
            and is_adjacent(placed_wildlife[0], pos)
        ):
            adjacent_fox_positions.add(placed_wildlife[0])

        adjacent_fox_reward = 0

        for fox_pos in adjacent_fox_positions:
            fox_adjacent_wildlife = {
                w for _, w in self.state.env.adjacent_wildlife(fox_pos)
            }

            if placed_wildlife is not None and is_adjacent(placed_wildlife[0], fox_pos):
                fox_adjacent_wildlife.add(placed_wildlife[1])

            if wildlife not in fox_adjacent_wildlife:
                adjacent_fox_reward += 1

        return adjacent_fox_reward

    @cached
    def _surrounding_empty(self, tile_position: HexPosition | None):
        surrounding_empty = {pos for pos in self.state.env.all_adjacent_empty()}
        if tile_position is not None:
            surrounding_empty.remove(tile_position)
            for apos in adjacent_positions(tile_position):
                if apos not in self.state.env.tiles:
                    surrounding_empty.add(apos)
        return surrounding_empty

    @cached
    def _best_adjacent_wreward(
        self,
        wildlife: Wildlife,
        placed_wildlife: tuple[HexPosition, Wildlife] | None,
        placed_tile_position: HexPosition | None,
    ):
        best_wreward = 0
        for pos in self._surrounding_empty(placed_tile_position):
            base_wreward = self._base_wreward(pos, wildlife, placed_wildlife)
            if base_wreward is not None:
                wreward = base_wreward + self._adjacent_fox_wreward(
                    pos, wildlife, placed_wildlife
                )
                best_wreward = max(wreward, best_wreward)
        return best_wreward

    @cached
    def _unoccupied(
        self,
        placed_tile: tuple[HexPosition, int] | None,
        placed_wildlife_position: HexPosition | None,
    ):
        if placed_tile is None or placed_wildlife_position == placed_tile[0]:
            return set(self.state.env.unoccupied_tiles())

        unoccupied = {
            (pos, tile)
            for pos, tile in self.state.env.unoccupied_tiles()
            if pos != placed_wildlife_position
        }
        unoccupied.add((placed_tile[0], self.state.tile_display[placed_tile[1]]))
        return unoccupied

    @cached
    def _best_unoccupied_wreward(
        self,
        wildlife: Wildlife,
        placed_wildlife: tuple[HexPosition, Wildlife] | None,
        placed_tile: tuple[HexPosition, int] | None,
    ):
        best_wreward = 0
        unoccupied = self._unoccupied(
            placed_tile, placed_wildlife[0] if placed_wildlife is not None else None
        )
        for pos, tile in unoccupied:
            if wildlife in tile.wildlife_slots:
                base_wreward = self._base_wreward(pos, wildlife, placed_wildlife)
                if base_wreward is not None:
                    wreward = (
                        int(tile.nature_token_reward)
                        + base_wreward
                        + self._adjacent_fox_wreward(pos, wildlife, placed_wildlife)
                    )
                    best_wreward = max(wreward, best_wreward)
        return best_wreward

    @cached
    def _habitat_groups(
        self,
        placed_tile: tuple[HexPosition, int, int] | None,
    ) -> dict[Habitat, list[set[HexPosition]]]:
        if placed_tile is None:
            return self.state.env.habitat_groups()

        hgroups = dict(self._habitat_groups(None))
        pos, index, rot = placed_tile
        tile = self.state.tile_display[index].rotate(rot)

        for h in set(tile.habitats):
            connected_positions = {
                apos
                for apos, atile in self.state.env.adjacent_tiles(pos)
                if share_edge(pos, tile, apos, atile, h)
            }

            new_groups = [{pos}]
            for group in hgroups[h]:
                if group.isdisjoint(connected_positions):
                    new_groups.append(group)
                else:
                    new_groups[0].update(group)
            hgroups[h] = sorted(new_groups, key=len, reverse=True)

        return hgroups

    @cached
    def _hreward(
        self,
        pos: HexPosition,
        habitat: Habitat,
        placed_tile: tuple[HexPosition, int, int] | None,
    ):
        groups = self._habitat_groups(placed_tile)[habitat]
        largest_group_size = len(groups[0])

        adjacent_tiles = list(self.state.env.adjacent_tiles(pos))
        if placed_tile is not None and is_adjacent(pos, placed_tile[0]):
            adjacent_tiles.append(
                (
                    placed_tile[0],
                    self.state.tile_display[placed_tile[1]].rotate(placed_tile[2]),
                )
            )

        dummy_tile = Tile((habitat, habitat), frozenset())

        connected_positions = {
            apos
            for apos, atile in adjacent_tiles
            if share_edge(pos, dummy_tile, apos, atile, habitat)
        }

        new_group_size = 1

        for group in groups:
            if not group.isdisjoint(connected_positions):
                new_group_size += len(group)

        hreward = 0

        if new_group_size > largest_group_size:
            hreward += new_group_size - largest_group_size
            if new_group_size >= 7 and not (largest_group_size >= 7):
                hreward += 2

        return hreward

    @cached
    def _best_hreward(
        self,
        habitat: Habitat,
        placed_tile: tuple[HexPosition, int, int] | None,
    ):
        best_hreward = 0
        for pos in self._surrounding_empty(
            placed_tile[0] if placed_tile is not None else None
        ):
            hreward = self._hreward(pos, habitat, placed_tile)
            best_hreward = max(hreward, best_hreward)
        return best_hreward

    @cached
    def _largest_area(
        self,
        habitat: Habitat,
        placed_tile: tuple[HexPosition, int, int] | None,
    ):
        groups = self._habitat_groups(placed_tile)[habitat]
        largest_group_size = len(groups[0])

        if placed_tile is None:
            return largest_group_size

        pos, index, rot = placed_tile
        tile = self.state.tile_display[index].rotate(rot)
        connected_positions = {
            apos
            for apos, atile in self.state.env.adjacent_tiles(pos)
            if share_edge(pos, tile, apos, atile, habitat)
        }

        new_group_size = 1
        for group in groups:
            if not group.isdisjoint(connected_positions):
                new_group_size += len(group)

        return max(new_group_size, largest_group_size)

    @cached
    def _available_positions(
        self, tile_position: HexPosition | None, wildlife_position: HexPosition | None
    ):
        all_tile_positions = set(self.state.env.tiles)
        if tile_position is not None:
            all_tile_positions.add(tile_position)

        all_adjacent_empty = {
            apos for tp in all_tile_positions for apos in adjacent_positions(tp)
        }

        available_positions = all_tile_positions | all_adjacent_empty
        available_positions -= set(self.state.env.wildlife)
        if wildlife_position is not None:
            available_positions.remove(wildlife_position)

        return available_positions

    def get_features(
        self, data: NDArray[np.float32] | None = None, action: Action | None = None
    ):
        if data is None:
            data = np.zeros(len(feature_names), dtype=np.float32)

        s = self.state

        data[F["turns_remaining"]] = (
            s.turns_remaining if action is None else s.turns_remaining - 1
        )

        data[F["nature_tokens"]] = self._nature_tokens(action)

        placed_wildlife = (
            None
            if (action is None or action.wildlife_position is None)
            else (
                action.wildlife_position,
                self.state.wildlife_display[action.wildlife_index],
            )
        )

        bsizes = Counter(
            len(g) for g in self._wildlife_groups(Wildlife.BEAR, placed_wildlife)
        )
        data[F["num_bear_pairs"]] = bsizes[2]
        data[F["num_bear_singles"]] = bsizes[1]

        data[F["num_elk"]] = self._wildlife_count(Wildlife.ELK, placed_wildlife)
        data[F["num_salmon"]] = self._wildlife_count(Wildlife.SALMON, placed_wildlife)
        data[F["num_hawks"]] = self._wildlife_count(Wildlife.HAWK, placed_wildlife)
        data[F["num_foxes"]] = self._wildlife_count(Wildlife.FOX, placed_wildlife)

        available_positions = self._available_positions(
            action.tile_position if action is not None else None,
            action.wildlife_position if action is not None else None,
        )

        placed_tile = (
            None
            if action is None
            else (action.tile_position, action.tile_index, action.tile_rotation)
        )

        for h in Habitat:
            data[F[f"{h.value}_largest_area"]] = self._largest_area(h, placed_tile)

        all_pos_features = []

        for pos in available_positions:
            tile = self.state.env.tiles.get(pos)
            if (
                action is not None
                and pos == action.tile_position
                and pos != action.wildlife_position
            ):
                tile = self.state.tile_display[action.tile_index]

            all_pos_features.append(
                [
                    *(
                        (0 if tile is None else int(w in tile.wildlife_slots))
                        for w in Wildlife
                    ),
                    *(
                        (
                            0
                            if tile is not None and w not in tile.wildlife_slots
                            else self._base_wreward(pos, w, placed_wildlife) or 0
                        )
                        for w in Wildlife
                    ),
                    *(
                        (0 if tile is not None else self._hreward(pos, h, placed_tile))
                        for h in Habitat
                    ),
                ]
            )

            # print(pos, pos_features)

        while len(all_pos_features) < 37:
            all_pos_features.append([0] * len(all_pos_features[0]))

        pos_features_array = np.array(all_pos_features, dtype=np.float32)

        return pos_features_array, data

    def get_next_features(self, actions: list[Action]):
        data = np.zeros((len(actions), len(feature_names)), dtype=np.float32)
        for i, action in enumerate(actions):
            self.get_features(data[i, :], action)
        return data

    def __repr__(self):
        data = self.get_features()
        return "\n".join(
            [
                "StateFeatures({",
                *(f"  '{fn}': {data[F[fn]]}," for fn in feature_names),
                "})",
            ]
        )


# # %%
# from cascadia_ai.ai.training_data import get_greedy_played_games

# greedy_played_games = get_greedy_played_games()


# # %%
# from cascadia_ai.tui import print_state
# from random import choice
# from cascadia_ai.ai.actions import get_actions_and_rewards
# import pandas as pd

# s = choice(greedy_played_games)[0]
# print_state(s)

# sf = StateFeatures(s)
# sf.get_features()


# # %%

# actions, rewards = get_actions_and_rewards(s)

# next_features = sf.get_next_features(actions)

# df = pd.DataFrame(next_features, columns=feature_names)

# stats = df.describe().T[["mean", "min", "max"]]
# stats["original"] = sf.get_features()

# stats
