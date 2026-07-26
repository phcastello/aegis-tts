from __future__ import annotations
from abc import ABC, abstractmethod
class StrictSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, representation, parameters, seed, duration_samples): ...
