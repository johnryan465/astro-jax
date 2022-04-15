from abc import ABC, abstractmethod
from astrojax.physics.actuators.actuators import Actuator

from astrojax.physics.bodies import Body
from astrojax.physics.forces import ForceConfig
from astrojax.state import PosVel
from astrojax.state import TimeDerivatives

import jumpy as jp

from astrojax import pytree
from typing import List, Tuple
from abc import ABC


@pytree.register
class Thruster(Actuator):
    """Applies a thrust to a body."""

    def __init__(self, forces: List[ForceConfig], body: Body, act_index: List[Tuple[int, int]]):
        self.body = jp.take(body, [body.index[f.body] for f in forces])
        self.strength = jp.array([f.strength for f in forces])
        self.act_index = jp.array(act_index)

    def apply_reduced(self, force: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        dvel = force * self.strength / self.body.mass
        return dvel, jp.zeros_like(dvel)

    def apply(self, qp: PosVel, force_data: jp.ndarray) -> TimeDerivatives:
        """Applies a force to a body.
        Args:
          qp: State data for system
          force_data: Data specifying the force to apply to a body.
        Returns:
          dP: The impulses that result from apply a force to the body.
        """

        force_data = jp.take(force_data, self.act_index)
        dvel, dang = jp.vmap(type(self).apply_reduced)(self, force_data)

        # sum together all impulse contributions to all parts effected by forces
        dvel = jp.segment_sum(dvel, self.body.idx, qp.pos.shape[0])
        dang = jp.segment_sum(dang, self.body.idx, qp.pos.shape[0])

        return TimeDerivatives(vel=dvel, ang=dang)
