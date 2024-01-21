from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd


class Agent(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def get_action(self, epsilon: float, device: str) -> Optional[int]:
        ...

    @abstractmethod
    def get_history() -> pd.DataFrame:
        ...
