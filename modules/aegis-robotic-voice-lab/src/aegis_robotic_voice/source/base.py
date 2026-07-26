from __future__ import annotations
from abc import ABC, abstractmethod
class SourceArticulator(ABC):
    @abstractmethod
    def generate(self, text: str, **parameters): ...
