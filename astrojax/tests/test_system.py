from astrojax.physics.integrator import Integrator
from astrojax.physics.system import System, SystemConfig
from astrojax.state.state import PosVel, Vel, Pos
import jax.numpy as jp


def test_correct():
    config = SystemConfig(dt=0.01, substeps=2, num_bodies=1)
    system = System(config)
    state = PosVel.zero(shape=(1,))
    print(system.step(state=state, act=jp.zeros((1,))))
