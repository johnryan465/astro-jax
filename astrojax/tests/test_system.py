from astrojax.physics.actuators import Thruster
from astrojax.physics.bodies import Body, BodyConfig
from astrojax.physics.forces import ForceConfig
from astrojax.physics.forces.gravity import Gravity
from astrojax.physics.integrator import Integrator
from astrojax.physics.system import System, SystemConfig
from astrojax.state.state import PosVel, TimeDerivatives, Pos
import jumpy as jp


def test_system():
    num_bodies = 1
    config = SystemConfig(dt=0.01, substeps=2, num_bodies=num_bodies)
    force = ForceConfig(strength=jp.ones(shape=(3)), body="spaceship")
    body = Body([
        BodyConfig(
            mass=1.0,
            inertia=jp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            name="spaceship",
            frozen=False
        )
    ])
    thruster = Thruster([force], body, [(0, 1, 2)])
    gravity = Gravity()
    system = System(config, [thruster], [gravity])
    state = PosVel.zero(shape=(num_bodies,))
    print(system.step(state=state, act=jp.ones((3, ))))
