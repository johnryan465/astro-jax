from abc import ABC, abstractmethod

from dataclasses import dataclass

from astrojax.state.state import PosVel, TimeDerivatives
import jumpy as jp


@dataclass
class LinkageConfig:
    parent_name: str
    child_name: str
    parent_idx: int
    child_idx: int



class Linkage(ABC):
    """
    Linkage is inspired by Brax's joints, however we wish to generalise to cases such as gravity.
    """
    
    @abstractmethod
    def apply(self, state: PosVel) -> TimeDerivatives:
        pass


