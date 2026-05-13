"""Persistent binary disk cache for API responses.

Cache key is SHA256 hash of ``endpoint|model|extract_reasoning|max_tokens|prompt``
where ``prompt`` = system + NUL-joined message contents.

Loaded once at startup via pickle, saved on normal exit, SIGTERM, or Ctrl+C.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BenchmarkConfig, ModelAPI

CACHE_PATH = os.environ.get(
    "BASIN_CACHE",
    os.path.expanduser("~/.cache/basin/responses.pkl"),
)


class ResponseCache:
    """Persistent binary cache keyed by prompt hash."""

    def __init__(self, path: str = CACHE_PATH, verbose: bool = True) -> None:
        self._path = path
        self._dirty = False
        self._verbose = verbose
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
            ):
                self._data = raw
            else:
                self._data = {}
            if self._verbose:
                print(
                    f"  cache: loaded {len(self._data)} entries from {self._path}",
                    file=sys.stderr,
                )
        except (FileNotFoundError, pickle.UnpicklingError, EOFError, OSError):
            self._data = {}
            if self._verbose:
                print(
                    f"  cache: no cache file at {self._path}, starting fresh",
                    file=sys.stderr,
                )

    def save(self) -> None:
        if not self._dirty:
            return
        with open(self._path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._dirty = False
        if self._verbose:
            print(
                f"  cache: saved {len(self._data)} entries to {self._path}",
                file=sys.stderr,
            )

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self._dirty = True

    def __len__(self) -> int:
        return len(self._data)


class CachedAPI:
    """Proxy around a ModelAPI that caches complete() responses to disk.

    Satisfies the ModelAPI protocol so it can replace the raw API anywhere.
    """

    def __init__(
        self, api: ModelAPI, cache: ResponseCache, config: BenchmarkConfig
    ) -> None:
        self._api = api
        self._cache = cache
        self._base_url = config.base_url or ""
        self._model = config.model or ""
        self._extract_reasoning = config.extract_reasoning
        self._hits = 0
        self._misses = 0
        self._skipped = 0

    def _key(
        self,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> str:
        prompt = system + "\0" + "\0".join(m["content"] for m in messages)
        raw = (
            f"{self._base_url}|{self._model}|{self._extract_reasoning}"
            f"|{max_tokens}|{prompt}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
    ) -> str:
        key = self._key(max_tokens, system, messages)
        cached = self._cache.get(key)
        if cached:
            self._hits += 1
            return cached
        self._misses += 1
        response = self._api.complete(system, messages, max_tokens)
        if response:
            self._cache.put(key, response)
        else:
            self._skipped += 1
        return response

    def count_tokens(self, text: str) -> int:
        return self._api.count_tokens(text)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def skipped(self) -> int:
        return self._skipped
