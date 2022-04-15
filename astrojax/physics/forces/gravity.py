from astrojax.physics.forces.forces import Force
from astrojax.state.state import PosVel, TimeDerivatives
import jumpy as jp


class Gravity(Force):
    def __init__(self, g: float = -9.81):
        self.gravity = jp.array([0, 0, g])

    def apply(self, state: PosVel) -> TimeDerivatives:
        return TimeDerivatives(vel=self.gravity, ang=jp.zeros_like(self.gravity))
