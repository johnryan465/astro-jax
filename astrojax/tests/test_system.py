from astrojax.physics.actuators import Thruster
from astrojax.physics.bodies import Body, BodyConfig
from astrojax.physics.integrator import Integrator
from astrojax.physics.system import System, SystemConfig
from astrojax.state.state import PosVel, TimeDerivatives, Pos
import jax.numpy as jp
import jax


@jax.jit
def test_correct():
    config = SystemConfig(dt=0.01, substeps=2, num_bodies=1)
    body = Body([
        BodyConfig(
            mass=1,
            inertia=jp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            name="spaceship",
            frozen=False
        )
    ])
    thruster = Thruster([], body, [])
    system = System(config, [thruster])
    state = PosVel.zero(shape=(1,))
    print(system.step(state=state, act=jp.zeros((1,))))
