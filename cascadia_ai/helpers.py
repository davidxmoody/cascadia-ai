from typing import TypeVar
from random import Random


T = TypeVar("T")


def shuffled(items: list[T], rand: Random | None = None) -> list[T]:
    rand = rand or Random()
    items_copy = items[:]
    rand.shuffle(items_copy)
    return items_copy
