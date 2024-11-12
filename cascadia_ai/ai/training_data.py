import pickle
from random import choice, choices, random
import torch
from tqdm import tqdm
from cascadia_ai.ai.actions import get_actions_and_rewards
from cascadia_ai.ai.features import get_next_features
from cascadia_ai.enums import Wildlife
from cascadia_ai.game_state import Action, GameState
from cascadia_ai.score import Score


# %%
single_bear_bonus = 3


def adjust_reward(state: GameState, action: Action, reward: int):
    wildlife = state.wildlife_display[action.wildlife_index]
    placed_single_bear = (
        wildlife == Wildlife.BEAR
        and action.wildlife_position is not None
        and not state.env.has_adjacent_wildlife(action.wildlife_position, Wildlife.BEAR)
    )
    return reward + single_bear_bonus if placed_single_bear else reward


def play_game_greedy_epsilon_biased(epsilon: float, until_turns_remaining: int = 0):
    state = GameState()

    while state.turns_remaining > until_turns_remaining:
        actions, rewards = get_actions_and_rewards(state)

        if random() < epsilon:
            state.take_action(choice(actions))

        else:
            rewards = [adjust_reward(state, a, r) for a, r in zip(actions, rewards)]

            max_reward = max(rewards)
            max_actions = [a for a, r in zip(actions, rewards) if r == max_reward]
            max_actions_no_nt = [a for a in max_actions if not a.nt_spent]

            action = choice(
                max_actions_no_nt if len(max_actions_no_nt) else max_actions
            )
            state.take_action(action)

    return state


def get_realistic_states(load=True) -> list[GameState]:
    if load:
        with open("data/realistic_states.pkl", "rb") as f:
            return pickle.load(f)

    realistic_states = []
    for i in tqdm(range(100000), desc=f"Generating realistic states"):
        realistic_states.append(play_game_greedy_epsilon_biased(0.1, i % 19 + 1))

    with open("data/realistic_states.pkl", "wb") as f:
        pickle.dump(realistic_states, f)

    return realistic_states


# %%
def play_game_greedy(state: GameState):
    while state.turns_remaining > 0:
        actions, rewards = get_actions_and_rewards(state)

        max_reward = max(rewards)
        max_actions = [a for a, r in zip(actions, rewards) if r == max_reward]
        max_actions_no_nt = [a for a in max_actions if not a.nt_spent]

        action = choice(max_actions_no_nt if len(max_actions_no_nt) else max_actions)
        state.take_action(action)

    return state


def get_greedy_played_games(load=True) -> list[tuple[GameState, list[Score]]]:
    if load:
        with open("data/greedy_played_games.pkl", "rb") as f:
            return pickle.load(f)

    realistic_states = get_realistic_states(load)
    greedy_played_games = []
    for state in tqdm(realistic_states, desc="Greedy playing games"):
        scores = [play_game_greedy(state.copy()).env.score for _ in range(5)]
        greedy_played_games.append((state, scores))

    with open("data/greedy_played_games.pkl", "wb") as f:
        pickle.dump(greedy_played_games, f)

    return greedy_played_games


# %%
def play_game(
    model,
    state: GameState | None = None,
    until_turns_remaining=0,
    gamma=0.9,
    epsilon=0.0,
):
    if state is None:
        state = GameState()

    while state.turns_remaining > until_turns_remaining:
        actions, rewards = get_actions_and_rewards(state)

        if state.turns_remaining == 1:
            i = rewards.index(max(rewards))

        elif epsilon and random() < epsilon:
            i = choices(range(len(actions)), weights=rewards, k=1)[0]

        else:
            next_features = [
                torch.from_numpy(nparray)
                for nparray in get_next_features(state, actions)
            ]

            with torch.no_grad():
                q_values = model(*next_features).squeeze()
            expected_rewards = q_values * gamma + torch.tensor(rewards)
            i = expected_rewards.argmax().item()

        state.take_action(actions[i])

    return state


def get_model_realistic_states(model, load=True) -> list[GameState]:
    if load:
        with open("data/model_realistic_states.pkl", "rb") as f:
            return pickle.load(f)

    realistic_states = []
    for i in tqdm(range(100000), desc=f"Generating model realistic states"):
        realistic_states.append(
            play_game(model, epsilon=0.1, until_turns_remaining=i % 19 + 1)
        )

    with open("data/model_realistic_states.pkl", "wb") as f:
        pickle.dump(realistic_states, f)

    return realistic_states


def get_model_played_games(model, load=True) -> list[tuple[GameState, list[Score]]]:
    if load:
        with open("data/model_played_games.pkl", "rb") as f:
            return pickle.load(f)

    realistic_states = get_model_realistic_states(load)
    model_played_games = []
    for state in tqdm(realistic_states, desc="Model playing games"):
        scores = [play_game(model, state=state.copy()).env.score for _ in range(5)]
        model_played_games.append((state, scores))

    with open("data/model_played_games.pkl", "wb") as f:
        pickle.dump(model_played_games, f)

    return model_played_games
