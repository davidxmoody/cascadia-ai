from rich.console import Console
from rich.theme import Theme
from cascadia_ai.game_state import GameState
from cascadia_ai.environment import Environment

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


def print_env(env: Environment):
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
                        char = qQ[0]
                    case "Q":
                        char = qQ[1]
                    case "r":
                        char = rR[0]
                    case "R":
                        char = rR[1]
                    case ",":
                        char = ","
                    case "A":
                        char = ABC[0]
                    case "B":
                        char = ABC[1]
                    case "C":
                        char = ABC[2]
                    case _:
                        char = " "

                chars[p] = f"[{style}]{char}[/{style}]"

    for y in range(min(p[1] for p in chars), max(p[1] for p in chars) + 1):
        line = ""
        for x in range(min(p[0] for p in chars), max(p[0] for p in chars) + 1):
            p = (x, y)
            line += chars[p] if p in chars else " "
        console.print(line, highlight=False)


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
