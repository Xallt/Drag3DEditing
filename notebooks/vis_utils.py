import plotly.graph_objs as go
import numpy as np

def pointcloud_trace(p):
    # Assume your point cloud is a NumPy array named 'point_cloud' with shape (N, 3)
    N, _ = p.shape

    # Create a trace for the 3D scatter plot
    trace = go.Scatter3d(
        x=p[:, 0],
        y=p[:, 1],
        z=p[:, 2],
        mode='markers',
        marker=dict(
            size=2,  # Adjust the marker size as needed
            color=np.arange(N),  # Assign a unique color to each point
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8  # Adjust the opacity of the markers
        )
    )
    return trace