from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from echoforge.contracts.domain import Hypothesis


class StreamingRecognizer(Protocol):
    model_id: str

    def accept_audio(self, samples: NDArray[np.float32], sample_rate: int) -> Hypothesis | None: ...

    def finalize(self) -> Hypothesis: ...

    def reset(self) -> None: ...


class EndpointFinalizer(Protocol):
    model_id: str

    def transcribe(self, samples: NDArray[np.float32], sample_rate: int) -> Hypothesis: ...
