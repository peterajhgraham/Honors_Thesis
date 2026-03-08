"""
3-D Loss Surface Visualisation

Renders an interactive 3-D surface plot using Plotly to illustrate the
landscape that gradient descent must navigate. The surface combines
quadratic and sinusoidal terms to produce a non-trivial optimisation
landscape with saddle points and local minima.

Functions
---------
visualize_3d_surface : Create and display a 3-D Plotly surface plot.
"""

import numpy as np
import plotly.graph_objects as go


def visualize_3d_surface():
    """Display an interactive 3-D surface of a mixed quadratic-sinusoidal function."""
    x_vals = np.linspace(-2, 2, 50)
    y_vals = np.linspace(-2, 2, 50)
    x, y = np.meshgrid(x_vals, y_vals)

    z = x**2 - y**2 + np.sin(3 * x) * np.cos(3 * y)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])

    fig.update_layout(
        title="Gradient Descent Visualisation",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        scene_camera=dict(up=dict(x=0, y=0, z=1)),
    )

    fig.show()
