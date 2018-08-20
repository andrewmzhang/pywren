import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as FF

import pandas as pd

traces= []

for i in range(1, len(sys.argv)):
    df = pd.read_csv(argv[i])

    trace = go.Scatter(
        x=df.iloc[:, 0],
        y=-df.iloc[:, 1],
        name=argv[i]
    )
    traces.append(trace)

fig = go.Figure(data=traces)


py.image.save_as(fig, filename="fig.png")

