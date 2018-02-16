# plotly chart for a notebook

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.offline as py
py.init_notebook_mode(connected=True)

data = [
    go.Heatmap(
        z= x_train.astype(float).corr().values ,
        x=x_train.columns.values,
        y= x_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')