from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class Registry[T]:
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._items:
            raise ValueError(f"Registry entry already exists: {name}")
        self._items[name] = factory

    def build(self, name: str, *args: object, **kwargs: object) -> T:
        try:
            factory = self._items[name]
        except KeyError as err:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise ValueError(f"Unknown registry entry {name!r}. Available: {available}") from err
        return factory(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)
