from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from astrojax.physics.actuators import Actuator
from astrojax.physics.forces import Force
from astrojax.physics.integrator import Integrator
import jumpy as jp

from astrojax.state import Pos, PosVel, TimeDerivatives
from astrojax import pytree


@dataclass
class SystemConfig:
    dt: float
    substeps: int
    num_bodies: int


@pytree.register
class System:
    def __init__(self, config: SystemConfig, actuators: List[Actuator]) -> None:
        self.config = config
        self.integrator = Integrator(self.config.dt)
        self.actuators = actuators
        self.forces = []
        self.joints = []

    def step(self, state: PosVel, act: jp.ndarray) -> PosVel:
        def substep(carry, _):
            qp, info = carry

            zero = TimeDerivatives.zero(shape=(self.config.num_bodies,))

            dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
            dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
            dp_j = sum([j.damp(qp) for j in self.joints], zero)
            qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_j)

            qp = self.integrator.kinetic(qp)
            return (qp, info), ()
        info = 0

        (state, info), _ = jp.scan(
            substep,
            (state, info),
            (),
            self.config.substeps // 2)
        return state
