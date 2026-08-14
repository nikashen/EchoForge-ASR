from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


class AudioRingBuffer:
    """Bounded float32 audio buffer with deterministic sample accounting."""

    def __init__(self, capacity_samples: int) -> None:
        if (
            isinstance(capacity_samples, bool)
            or not isinstance(capacity_samples, int)
            or capacity_samples < 1
        ):
            raise ValueError("capacity_samples must be a positive integer")
        self.capacity_samples = capacity_samples
        self._chunks: deque[NDArray[np.float32]] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._chunks.clear()
        self._size = 0

    def append(self, samples: NDArray[np.float32]) -> None:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("ring-buffer audio must be finite one-dimensional samples")
        if array.size >= self.capacity_samples:
            self._chunks.clear()
            self._chunks.append(np.ascontiguousarray(array[-self.capacity_samples :]))
            self._size = self.capacity_samples
            return
        self._chunks.append(np.ascontiguousarray(array.copy()))
        self._size += array.size
        while self._size > self.capacity_samples:
            overflow = self._size - self.capacity_samples
            first = self._chunks[0]
            if first.size <= overflow:
                self._chunks.popleft()
                self._size -= first.size
            else:
                self._chunks[0] = np.ascontiguousarray(first[overflow:])
                self._size -= overflow

    def to_array(self) -> NDArray[np.float32]:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(tuple(self._chunks)).astype(np.float32, copy=False)
