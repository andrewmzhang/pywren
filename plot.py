import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as FF

import pandas as pd


df = pd.read_csv('test_error')

trace = go.Scatter(
        x=df.iloc[:, 0],
        y=-df.iloc[:, 1]
        )

fig = go.Figure(data=[trace])

#offline.plot(fig, image="png")

py.image.save_as(fig, filename="fig.png")

