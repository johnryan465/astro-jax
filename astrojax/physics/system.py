from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from astrojax.physics.integrator import Integrator
import jax
import jax.numpy as jp

from astrojax.state import Pos, PosVel, Vel


@dataclass
class SystemConfig:
    dt: float
    substeps: int
    num_bodies: int


class System:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.integrator = Integrator(self.config.dt)
        self.actuators = []
        self.forces = []
        self.joints = []

    def step(self, state: PosVel, act: jp.ndarray) -> PosVel:
        def substep(carry, _):
            qp, info = carry

            zero = Vel(jp.zeros((self.config.num_bodies, 3)), jp.zeros((self.config.num_bodies, 3)))

            dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
            dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
            dp_j = sum([j.damp(qp) for j in self.joints], zero)
            qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_j)

            qp = self.integrator.kinetic(qp)
            return (qp, info), ()
        info = 0

        (state, info), _ = jax.lax.scan(
            substep,
            (state, info),
            (),
            self.config.substeps // 2)
        return state
