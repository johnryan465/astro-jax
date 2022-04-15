from abc import ABC, abstractmethod

from astrojax.state import PosVel
from astrojax.state import TimeDerivatives

from abc import ABC


class Force(ABC):
    @abstractmethod
    def apply(self, state: PosVel) -> TimeDerivatives:
        pass
