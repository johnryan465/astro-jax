from dataclasses import dataclass
from typing import List

from astrojax.state.state import PosVel, TimeDerivatives
import jumpy as jp
from brax import pytree


@dataclass
class BodyConfig:
    mass: float
    inertia: jp.ndarray
    name: str
    frozen: bool


@pytree.register
class Body:
    """
    A body is a solid, non-deformable object with some mass and shape.
    Attributes:
      idx: Index of where body is found in the system.
      inertia: (3, 3) Inverse Inertia matrix represented in body frame.
      mass: Mass of the body.
      active: whether the body is effected by physics calculations
      index: name->index dict for looking up body names
    """
    __pytree_ignore__ = ('index',)

    def __init__(self, bodies: List[BodyConfig]):
        self.idx = jp.arange(0, len(bodies))
        self.inertia = jp.inv(jp.array([b.inertia for b in bodies]))
        self.mass = jp.array([b.mass for b in bodies])
        self.active = jp.array([0.0 if b.frozen else 1.0 for b in bodies])
        self.index = {b.name: i for i, b in enumerate(bodies)}

    def impulse(self, qp: PosVel, impulse: jp.ndarray, pos: jp.ndarray) -> TimeDerivatives:
        """Calculates updates to state information based on an impulse.
        Args:
          qp: State data of the system
          impulse: Impulse vector
          pos: Location of the impulse relative to the body's center of mass
        Returns:
          dP: An impulse to apply to this body
        """
        dvel = impulse / self.mass
        dang = self.inertia * jp.cross(pos - qp.pos, impulse)
        return TimeDerivatives(vel=dvel, ang=dang)
