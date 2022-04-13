from astrojax.physics.integrator import Integrator
from astrojax.state.state import PosVel, Vel, Pos


def test_correct():
    state = PosVel.zero(shape=(1,))
    integrator = Integrator(0.01)
    state = integrator.update(state, acc_p=Vel.zero(shape=(1,)))
    print(integrator.kinetic(state))
