from astrojax.physics.integrator import Integrator
from astrojax.state.state import PosVel, TimeDerivatives, Pos


def test_correct():
    state = PosVel.zero(shape=(1,))
    integrator = Integrator(0.01)
    state = integrator.update(state, acc_p=TimeDerivatives.zero(shape=(1,)))
    print(integrator.kinetic(state))
