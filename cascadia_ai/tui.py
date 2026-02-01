from typing import NamedTuple

from rich.console import Console
from rich.theme import Theme
from cascadia_ai.game_state import GameState
from cascadia_ai.environment import Environment
from cascadia_ai.score import Score


class Bounds(NamedTuple):
    min_x: int
    max_x: int
    min_y: int
    max_y: int


color_template = [
    r"    0    ",
    r"  55000  ",
    r" 4455011 ",
    r" 44XXX11 ",
    r"  33322  ",
    r"    3    ",
]

text_template = [
    r"         ",
    r"         ",
    r"  qQ,rR  ",
    r"   ABC   ",
    r"         ",
    r"         ",
]

console = Console(
    force_terminal=True,
    theme=Theme(
        {
            "hm": "black on #7A7A7A",
            "hf": "white on #1B3F16",
            "hp": "black on #F0C846",
            "hw": "black on #AACD64",
            "hr": "black on #4999BE",
            "wempty": "black on #FFFFFF",
            "wb": "white on #524139",
            "we": "black on #B49A4E",
            "ws": "black on #DC4C55",
            "wh": "black on #7DC0E6",
            "wf": "black on #D87334",
        }
    ),
)


def print_env(
    env: Environment, bounds: Bounds | None = None, show_coords=False
) -> Bounds:
    chars: dict[tuple[int, int], str] = {}

    for (q, r), tile in env.tiles.items():
        hl, hr = tile.habitats
        sides = [hr, hr, hr, hl, hl, hl]

        x = q * 8 + r * 4
        y = r * -4

        qQ = str(q).rjust(2)
        rR = str(r).ljust(2)

        if (w := env.wildlife.get((q, r), None)) is not None:
            ABC = f" {w.value.upper()} "
            wstyle = f"w{w.value}"
        else:
            ABC = "".join(sorted(ws.value for ws in tile.wildlife_slots)).center(3)
            wstyle = "wempty"

        for dy, line in enumerate(color_template):
            for dx, color_placeholder in enumerate(line):
                if color_placeholder == " ":
                    continue

                text_placeholder = text_template[dy][dx]
                p = (x + dx, y + dy)

                if color_placeholder.isdigit():
                    style = f"h{sides[int(color_placeholder) - tile.rotation].value.lower()}"
                else:
                    style = wstyle

                match text_placeholder:
                    case "q":
                        char = qQ[0] if show_coords else " "
                    case "Q":
                        char = qQ[1] if show_coords else " "
                    case "r":
                        char = rR[0] if show_coords else " "
                    case "R":
                        char = rR[1] if show_coords else " "
                    case ",":
                        char = "," if show_coords else " "
                    case "A":
                        char = ABC[0]
                    case "B":
                        char = ABC[1]
                    case "C":
                        char = ABC[2]
                    case _:
                        char = " "

                chars[p] = f"[{style}]{char}[/{style}]"

    if bounds is None:
        bounds = Bounds(
            min_x=min(p[0] for p in chars),
            max_x=max(p[0] for p in chars),
            min_y=min(p[1] for p in chars),
            max_y=max(p[1] for p in chars),
        )

    for y in range(bounds.min_y, bounds.max_y + 1):
        line = ""
        for x in range(bounds.min_x, bounds.max_x + 1):
            p = (x, y)
            line += chars[p] if p in chars else " "
        console.print(line, highlight=False)

    return bounds


def print_score(score: Score):
    wildlife_parts = [
        f"[w{w.value}] {w.value.upper()}: {v:>2} [/w{w.value}]"
        for w, v in score.wildlife.items()
    ]
    console.print("Wildlife: " + " ".join(wildlife_parts), highlight=False)

    habitat_parts = [
        f"[h{h.value.lower()}] {h.value}: {v:>2} [/h{h.value.lower()}]"
        for h, v in score.habitat.items()
    ]
    console.print("Habitat:  " + " ".join(habitat_parts), highlight=False)

    console.print(f"Nature tokens: {score.nature_tokens}", highlight=False)
    console.print(f"Total score: {score.total}", highlight=False)


def print_state(state: GameState):
    print(f"Turns remaining: {state.turns_remaining}")
    print(f"Nature tokens: {state.env.nature_tokens}")
    print()

    for i, (t, w) in enumerate(zip(state.tile_display, state.wildlife_display)):
        print(f"{i}: {str(t).ljust(12)} / {w}")
    print()

    print_env(state.env)
    print()

    print(state.env.score)
    print()
