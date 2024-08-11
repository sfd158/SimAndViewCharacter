'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''

import numpy as np


class DrawHelper:
    def __init__(self):
        pass

    @staticmethod
    def matplotlib3d_format_coord(ax, xd, yd):
        """
        Given the 2D view coordinates attempt to guess a 3D coordinate.
        Looks for the nearest edge to the point and then assumes that
        the point is at the same z location as the nearest point on the edge.

        Modified from ax.format_coord(event.xdata, event.ydata)
        """
        from mpl_toolkits.mplot3d import proj3d

        if ax.M is None:
            return ''

        if ax.button_pressed in ax._rotate_btn:
            return 'azimuth={:.0f} deg, elevation={:.0f} deg '.format(
                ax.azim, ax.elev)
            # ignore xd and yd and display angles instead

        # nearest edge
        p0, p1 = min(ax.tunit_edges(), key=lambda edge: proj3d._line2d_seg_dist(edge[0], edge[1], (xd, yd)))

        # scale the z value to match
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        d0 = np.hypot(x0-xd, y0-yd)
        d1 = np.hypot(x1-xd, y1-yd)
        dt = d0+d1
        z = d1/dt * z0 + d0/dt * z1


        x, y, z = proj3d.inv_transform(xd, yd, z, ax.M)
        return x, y, z
