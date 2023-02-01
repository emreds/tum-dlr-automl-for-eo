import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches

from compare_acc_LCZ_cifar import getMeanLCZ, getMeanCIFAR10, getVarLCZ, getVarCIFIAR10


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

limits = [1.0, 0.01, 1.0, 1.0, 1.00, 1.0, 1.0, 1.00]
data = [
    ['Fitness - Mean', 'Fitness - Variance', 'Negative Persistance - AuC', 'Negative Persistance',
     'Positive Persistance - AuC', 'Positive Persistance', 'Ruggedness', 'Cardinal Of Optima'],
    ('Radar chart', [
        [getMeanLCZ(), getVarLCZ()/limits[1], 0.0, 0.0, 0.00, 0.0, 0.0, 0.00],
        [getMeanCIFAR10(), getVarCIFIAR10()/limits[1], 0.0, 0.0, 0.00, 0.0, 0.0, 0.00]])
]

N = len(data[0])
theta = radar_factory(N, frame='polygon')
print(theta)

spoke_labels = data.pop(0)
title, case_data = data[0]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)


ax.set_rgrids([0.25, 0.5, 0.75, 1],[0.25, 0.5, 0.75, 1],angle=theta[0]*180/np.pi)
ax.set_rgrids([0.25, 0.5, 0.75, 1],[0.25*limits[1], 0.5*limits[1], 0.75*limits[1], 1*limits[1]],angle=theta[1]*180/np.pi)
ax.set_title(title,  position=(0.5, 1.1), ha='center')

colors = ['orange','blue']


so2sat_patch = mpatches.Patch(color='orange', label='So2Sat LCZ42', alpha=0.25)
cifar_patch = mpatches.Patch(color='blue', label='CIFAR10', alpha=0.25)
plt.legend(handles=[so2sat_patch, cifar_patch])

for i, d in enumerate(case_data):
    line = ax.plot(theta, d,color=colors[i])
    ax.fill(theta, d,  alpha=0.25,facecolor=colors[i])
    # ax.set_ylim(0, limits[i])
ax.set_varlabels(spoke_labels)

plt.show()
