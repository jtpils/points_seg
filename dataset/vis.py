import numpy as np
import mayavi.mlab as mlab

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


def draw_lidar_simple(points, labels, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = labels * 30 + 20
    # draw points
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color, color=None, mode='point', scale_factor=20, colormap='jet',
                  figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=1)
    # draw axis
    axes = np.array([
        [5., 0., 0., 0.],
        [0., 5., 0., 0.],
        [0., 0., 5., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

