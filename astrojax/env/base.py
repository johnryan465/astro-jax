from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: float) -> None:
        pass

    @abstractmethod
    def draw(self) -> None:
        pass
