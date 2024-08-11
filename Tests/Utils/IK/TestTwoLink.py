import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from scipy.spatial.transform import Rotation
from VclSimuBackend.Common.Draw import DrawHelper
from VclSimuBackend.Utils.IK.TwoLinkIK import two_link_fk, two_link_ik

offset, local_quat, pos = np.array([]), np.array([]), np.array([])
lines, points = [], []


def onclick(event):
    global local_quat, pos
    if event.button == MouseButton.LEFT:
        pass
    elif event.button == MouseButton.MIDDLE:
        # coords = ax_global.format_coord(event.xdata, event.ydata)
        coords = np.array(DrawHelper.matplotlib3d_format_coord(event.inaxes, event.xdata, event.ydata))
        last_pos = pos[:, -1, :]
        print("Target Position is ", coords, "Last Position is", last_pos)
        local_quat = two_link_ik(local_quat, offset, coords[None, ...])
        _, _, pos = two_link_fk(local_quat, offset)
        print("Now Position is ", pos[0, -1, :])
        set_render_data(last_pos)
        event.canvas.draw()
        event.canvas.flush_events()
    elif event.button == MouseButton.RIGHT:
        pass


def set_render_data(last_pos: np.ndarray):
    lines[0].set_data_3d(pos[0, 0:2, 0], pos[0, 0:2, 1], pos[0, 0:2, 2])
    lines[1].set_data_3d(pos[0, 1:3, 0], pos[0, 1:3, 1], pos[0, 1:3, 2])
    points[0].set_data_3d(*last_pos)
    points[1].set_data_3d(*pos[0, -1, :])


def re_render(ax, last_pos: np.ndarray):
    set_render_data(last_pos)
    ax.set_xlim(np.min(pos[..., 0]) - 0.2, np.max(pos[..., 0]) + 0.2)
    ax.set_ylim(np.min(pos[..., 1]) - 0.2, np.max(pos[..., 1]) + 0.2)
    ax.set_zlim(np.min(pos[..., 2]) - 0.2, np.max(pos[..., 2]) + 0.2)


def test_two_link_ik():
    global offset, local_quat, pos, lines, points
    offset = np.array([[0, 0, 1], [0, 0.5, 0.5]], dtype=np.float64)
    local_quat = Rotation.random(2).as_quat()[None, ...]

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    ax = plt.axes(projection='3d')
    local_rot_arr, global_rot_arr, pos = two_link_fk(local_quat, offset)
    lines = [ax.plot([0, 0], [0, 0], [0, 0])[0] for _ in range(2)]
    points = [ax.plot(*pos[0, -1, :], marker='.', linestyle='None')[0] for _ in range(2)]
    re_render(ax, pos[:, -1, :])
    plt.show()


if __name__ == "__main__":
    test_two_link_ik()
