from astrojax.physics.actuators import Thruster
from astrojax.physics.bodies import Body, BodyConfig
from astrojax.physics.forces import Force
from astrojax.physics.integrator import Integrator
from astrojax.physics.system import System, SystemConfig
from astrojax.state.state import PosVel, TimeDerivatives, Pos
import jumpy as jp
import jax


def test_correct():
    num_bodies = 1
    config = SystemConfig(dt=0.01, substeps=2, num_bodies=num_bodies)
    force = Force(strength=jp.ones(shape=(3)), body="spaceship")
    body = Body([
        BodyConfig(
            mass=1.0,
            inertia=jp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            name="spaceship",
            frozen=False
        )
    ])
    thruster = Thruster([force], body, [(0, 1, 2)])
    system = System(config, [thruster])
    state = PosVel.zero(shape=(num_bodies,))
    print(system.step(state=state, act=jp.ones((3, ))))
