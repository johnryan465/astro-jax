from abc import ABC, abstractmethod


class Frame(ABC):
    """
    Frame defines a coordinate system
    """
    @abstractmethod
    def inertial(self) -> bool:
        pass
