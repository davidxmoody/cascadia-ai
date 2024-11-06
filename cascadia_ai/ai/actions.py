from cascadia_ai.enums import Wildlife
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.positions import HexPosition
from cascadia_ai.tiles import Tile


def pick_best_rotations(
    state: GameState,
    tile: Tile,
    pos: HexPosition,
):
    rotations = list(range(1 if tile.single_habitat else 6))
    rotated_tiles = [tile.rotate(rot) for rot in rotations]
    rewards = [
        sum(state.env.hlayers[h].get_tile_reward(pos, t) for h in t.unique_habitats)
        for t in rotated_tiles
    ]
    max_reward = max(rewards)
    return [
        (rot, reward) for rot, reward in zip(rotations, rewards) if reward == max_reward
    ]


def tile_options(state: GameState):
    for tpos in state.env.all_adjacent_empty():
        for tindex in range(4):
            tile = state.tile_display[tindex]
            for trot, treward in pick_best_rotations(state, tile, tpos):
                yield (tindex, tpos, trot, treward)


def windex_options(state: GameState, tindex: int):
    yield (tindex, 0)

    if state.env.nature_tokens > 0:
        considered_wildlife = {state.wildlife_display[tindex]}

        for i in range(4):
            if state.wildlife_display[i] not in considered_wildlife:
                considered_wildlife.add(state.wildlife_display[i])
                yield (i, -1)


def wpos_options(state: GameState, tindex: int, tpos: HexPosition, windex: int):
    wildlife = state.wildlife_display[windex]
    placed_tile = state.tile_display[tindex]

    if wildlife in placed_tile.wildlife_slots:
        yield (tpos, int(placed_tile.nature_token_reward))

    for pos, tile in state.env.unoccupied_tiles():
        if wildlife in tile.wildlife_slots:
            yield (pos, int(tile.nature_token_reward))


def calculate_wreward(state: GameState, pos: HexPosition, wildlife: Wildlife):
    reward = state.env.wlayers[wildlife].get_reward(pos, wildlife)

    if wildlife != Wildlife.FOX:
        reward += state.env.wlayers[Wildlife.FOX].get_reward(pos, wildlife) or 0

    return reward


def get_actions_and_rewards(state: GameState) -> tuple[list[Action], list[int]]:
    actions: list[Action] = []
    rewards: list[int] = []

    wcache: dict[tuple[Wildlife, HexPosition], int] = {}

    for tindex, tpos, trot, treward in tile_options(state):
        wplaced = False

        for windex, ntcost in windex_options(state, tindex):
            for wpos, ntreward in wpos_options(state, tindex, tpos, windex):
                wildlife = state.wildlife_display[windex]

                key = (wildlife, wpos)
                if key in wcache:
                    wreward = wcache[key]
                else:
                    wreward = calculate_wreward(state, wpos, wildlife)
                    wcache[key] = wreward

                if wreward >= 0:
                    wplaced = True
                    actions.append(Action(tindex, tpos, trot, windex, wpos))
                    rewards.append(treward + ntreward + ntcost + wreward)

        if not wplaced:
            actions.append(Action(tindex, tpos, trot, tindex, None))
            rewards.append(treward)

    return (actions, rewards)
