import pickle
import time
from cascadia_ai.tui import print_env, print_score, Bounds


def cursor_home():
    print("\033[H", end="")


def playback_game(states: list, delay: float = 0.5):
    final_env = states[-1].env

    chars_positions = []
    for (q, r) in final_env.tiles.keys():
        x = q * 8 + r * 4
        y = r * -4
        for dx in range(9):
            for dy in range(6):
                chars_positions.append((x + dx, y + dy))

    bounds = Bounds(
        min_x=min(p[0] for p in chars_positions),
        max_x=max(p[0] for p in chars_positions),
        min_y=min(p[1] for p in chars_positions),
        max_y=max(p[1] for p in chars_positions),
    )

    print(f"Board bounds: {bounds}")
    print(f"Playing {len(states)} states with {delay}s delay...")
    time.sleep(2)

    print("\033[2J", end="")

    for i, state in enumerate(states):
        cursor_home()
        print(f"Turn: {i + 1}/{20}" if i < 20 else "Turn: 20/20 (game ended)")
        print()
        print_score(state.env.score)
        print()
        print_env(state.env, bounds=bounds, show_coords=False)
        print()
        time.sleep(delay)

    time.sleep(5)



if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "cascadia_ai/demo/game.pkl"
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

    print(f"Loading states from {input_path}...")
    with open(input_path, "rb") as f:
        states = pickle.load(f)

    playback_game(states, delay=delay)
