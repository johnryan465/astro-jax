# %%
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt

from astrojax.env.two_body import TwoBody
import jumpy as jp

env = TwoBody()

print(env.step(action=jp.ones((3, ))))

print(env.state)
# %%

states = []
for i in range(1000):
    states.append(env.state)
    state = env.step(action=jp.ones((3, )))


# %%

print(list([s.vel[0][0] for s in states]))
# %%
plt.plot([s.pos[0][0] for s in states])
# %%
import plotly.express as px
import pandas as pd
x = [s.vel[0][0] for s in states]
y = [s.vel[1][0] for s in states]
z = [i for i in range(len(states))]


# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()

# Configure the trace.
df = pd.DataFrame(dict(x=x, y=y))
px.scatter(df)


# %%
# %matplotlib widget

x = [s.pos[0][0] for s in states]
y = [s.pos[1][0] for s in states]
z = [i for i in range(len(states))]


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

# %%
