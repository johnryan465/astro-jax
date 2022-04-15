from abc import ABC, abstractmethod

from astrojax.state import PosVel
from astrojax.state import TimeDerivatives

import jumpy as jp

from abc import ABC


class Actuator(ABC):
    @abstractmethod
    def apply(self, state: PosVel, actions: jp.ndarray) -> TimeDerivatives:
        pass
