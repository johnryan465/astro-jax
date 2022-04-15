# %%
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt
from astrojax.physics.actuators import Thruster
from astrojax.physics.bodies import Body, BodyConfig
from astrojax.physics.forces import ForceConfig
from astrojax.physics.forces.gravity import Gravity
from astrojax.physics.integrator import Integrator
from astrojax.physics.system import System, SystemConfig
from astrojax.state.state import PosVel, TimeDerivatives, Pos
import jumpy as jp


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
system = System(config, [], [gravity])
state = PosVel.zero(shape=(num_bodies,))
print(system.step(state=state, act=jp.ones((3, ))))

# %%

states = []
for i in range(100):
    states.append(state)
    state = system.step(state=state, act=jp.ones((3, )))


# %%

print(list([s.vel[0][0] for s in states]))
# %%
plt.plot([s.pos[0][0] for s in states])
# %%
# %matplotlib widget

x = [s.vel[0][0] for s in states]
y = [s.vel[0][1] for s in states]
z = [s.vel[0][2] for s in states]


# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()

# Configure the trace.
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker={
        'size': 10,
        'opacity': 0.8,
    }
)

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)
# %%
