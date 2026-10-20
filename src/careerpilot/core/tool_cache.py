"""A tiny session-scoped read cache for agent tool calls.

Tool-calling agents (Market, Analytics) often repeat the same lookup within one
conversation. This memoizes a tool call's result by ``(tool_name, arguments)`` for
the life of a single session so repeated reads are not re-fetched. Deliberately not
a global cache: finance data changes, and a new session should see fresh reads.
"""

from __future__ import annotations

import json
from typing import Any


def _key(tool_name: str, arguments: dict) -> str:
    # Sort keys so argument order never produces a cache miss for the same call.
    return tool_name + "::" + json.dumps(arguments, sort_keys=True, default=str)


class ToolCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, tool_name: str, arguments: dict) -> Any | None:
        return self._store.get(_key(tool_name, arguments))

    def set(self, tool_name: str, arguments: dict, result: Any) -> None:
        self._store[_key(tool_name, arguments)] = result

    def __len__(self) -> int:  # handy for assertions in tests
        return len(self._store)
