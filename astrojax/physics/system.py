from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from astrojax.physics.actuators import Actuator
from astrojax.physics.forces import ForceConfig
from astrojax.physics.forces.forces import Force
from astrojax.physics.integrator import Integrator
import jumpy as jp
from astrojax.physics.linkage.base import Linkage

from astrojax.state import Pos, PosVel, TimeDerivatives
from brax import pytree


@dataclass
class SystemConfig:
    dt: float
    substeps: int
    num_bodies: int


@pytree.register
class System:
    """
    This is a system which defines the physics of the system.
    
    Our system operates on a set of bodies, the state of these bodies are described their position, velocity, rotation and angular velocity.

    We apply forces to these bodies, and the system will update these states.
    Our actions are passed to "actuators" which lead to an additional set of forces/accelerations which are then applied to the bodies.
    Our system also contains linages which are used to constrain the movement of the bodies relative to each other, 
    we can effectively view these as forces in which 2 bodies can impact each other.

    """

    def __init__(self, config: SystemConfig, actuators: List[Actuator], forces: List[Force], linkages: List[Linkage]) -> None:
        self.config = config
        self.integrator = Integrator(self.config.dt, self.config.num_bodies)
        self.actuators = actuators
        self.forces = forces
        self.linkages = linkages

    def step(self, state: PosVel, act: jp.ndarray) -> PosVel:
        def substep(carry, _):
            qp, info = carry

            zero = TimeDerivatives.zero(shape=(self.config.num_bodies,))
            dp_l = sum([j.apply(qp) for j in self.linkages], zero)
            dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
            dp_f = sum([f.apply(qp) for f in self.forces], zero)
            qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_l)
            qp = self.integrator.kinetic(qp)
            return (qp, info), ()
        info = 0

        (state, info), _ = jp.scan(
            substep,
            (state, info),
            (),
            self.config.substeps)
        return state
