from typing import List
from gym import Space
from dataclasses import dataclass
from astrojax.env.base import Env
from astrojax.physics import actuators
from astrojax.physics.actuators.actuators import Actuator
from astrojax.physics.actuators.thruster import Thruster
from astrojax.physics.bodies import Body, BodyConfig
from astrojax.physics.forces.forces import Force
from astrojax.physics.forces.forces_config import ForceConfig
from astrojax.physics.linkage.base import Linkage
from astrojax.physics.linkage.gravity import GravityLinkageConfig, TwoBodyGravity
from astrojax.physics.system import System, SystemConfig
import jumpy as jp

from astrojax.state.state import PosVel

@dataclass
class Earth:
    mass: float = 10.0
    inertia: jp.ndarray = jp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

@dataclass
class Spacecraft:
    mass: float = 1.0
    inertia: jp.ndarray = jp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TwoBody(Env):
    def __init__(self) -> None:
        self.earth = Earth()
        self.spacecraft = Spacecraft()
        self.num_bodies = 2

        config = SystemConfig(dt=0.01, substeps=5, num_bodies=self.num_bodies)

        self.body = self.create_body()
        forces = self.create_forces()
        linkages = self.create_linkages()
        actuators = self.create_actuators()

        self.system = System(config, actuators=actuators, forces=forces, linkages=linkages)

        self.state = self.create_state() 
        super().__init__()

    

    def create_body(self) -> Body:
        earth_config = BodyConfig(
            mass=self.earth.mass,
            inertia=self.earth.inertia,
            name="earth",
            frozen=False
        )
        spacecraft_config = BodyConfig(
            mass=self.spacecraft.mass,
            inertia=self.spacecraft.inertia,
            name="spacecraft",
            frozen=False
        )
        return Body([
            earth_config,
            spacecraft_config
        ])

    def create_actuators(self) -> List[Actuator]:
        return []

    def create_forces(self) -> List[Force]:
        return []

    def create_state(self) -> PosVel:
        return PosVel(
            pos=jp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            vel=jp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ang=jp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            rot=jp.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def create_linkages(self) -> List[Linkage]:
        # The first linkage we create is gravity.
        gravity = TwoBodyGravity(
            body=self.body,
            linkages=[
                GravityLinkageConfig(
                    parent_name="earth",
                    child_name="spacecraft",
                    parent_mass=self.earth.mass,
                    child_mass=self.spacecraft.mass,
                    parent_idx=0,
                    child_idx=1,
                )
            ]

        )
        return [gravity]


    def step(self, action: jp.ndarray) -> PosVel:
        self.state = self.system.step(state=self.state, act=action)
        return self.state

    def reset(self) -> None:
        self.state = self.create_state()

    def draw(self) -> None:
        pass

    