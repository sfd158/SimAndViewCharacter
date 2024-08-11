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

"""
view euler angle range by sampling
"""
import sys
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QApplication, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton


class WindowClass(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super(WindowClass, self).__init__(parent)

        # x low high
        x_low_high = QHBoxLayout()
        x_low_label = QLabel()
        x_low_label.setText("x low")
        self.x_low_text = QLineEdit()
        self.x_low_text.setText("-180")
        x_low_high.addWidget(x_low_label)
        x_low_high.addWidget(self.x_low_text)

        x_high_label = QLabel()
        x_high_label.setText("x high")
        self.x_high_text = QLineEdit()
        self.x_high_text.setText("180")
        x_low_high.addWidget(x_high_label)
        x_low_high.addWidget(self.x_high_text)

        # y low high
        y_low_high = QHBoxLayout()
        y_low_label = QLabel()
        y_low_label.setText("y low")
        self.y_low_text = QLineEdit()
        self.y_low_text.setText("-180")
        y_low_high.addWidget(y_low_label)
        y_low_high.addWidget(self.y_low_text)

        y_high_label = QLabel()
        y_high_label.setText("y high")
        self.y_high_text = QLineEdit()
        self.y_high_text.setText("180")
        y_low_high.addWidget(y_high_label)
        y_low_high.addWidget(self.y_high_text)

        # z low high
        z_low_high = QHBoxLayout()
        z_low_label = QLabel()
        z_low_label.setText("z low")
        self.z_low_text = QLineEdit()
        self.z_low_text.setText("-180")
        z_low_high.addWidget(z_low_label)
        z_low_high.addWidget(self.z_low_text)

        z_high_label = QLabel()
        z_high_label.setText("z high")
        self.z_high_text = QLineEdit()
        self.z_high_text.setText("180")
        z_low_high.addWidget(z_high_label)
        z_low_high.addWidget(self.z_high_text)

        # euler angles
        self.euler_case = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]
        self.euler_order = QComboBox()
        self.euler_order.addItems(self.euler_case)

        self.draw_button = QPushButton()
        self.draw_button.setText("Draw")
        self.draw_button.clicked.connect(self.draw_button_func)

        self.view_step = QLineEdit()
        self.view_step.setText("10")

        euler_draw_layout = QHBoxLayout()
        euler_draw_layout.addWidget(self.euler_order)
        euler_draw_layout.addWidget(self.view_step)
        euler_draw_layout.addWidget(self.draw_button)

        xyz_angle = QVBoxLayout()
        xyz_angle.addLayout(x_low_high)
        xyz_angle.addLayout(y_low_high)
        xyz_angle.addLayout(z_low_high)
        xyz_angle.addLayout(euler_draw_layout)

        tot_layout = QHBoxLayout()
        # tot_layout.addWidget(self.canvas)
        tot_layout.addLayout(xyz_angle)

        self.setLayout(tot_layout)
        self.setWindowTitle("Euler angle to axis angle range")

    @staticmethod
    def cartesian_product_simple_transpose(arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[i, ...] = a

        return arr.reshape(la, -1).T

    def parse_euler_tmp(self):
        try:
            x_low = float(self.x_low_text.text())
        except ValueError:
            print("x low angle is not a number")
            return

        try:
            x_high = float(self.x_high_text.text())
        except ValueError:
            print("x high angle is not a number")
            return

        try:
            y_low = float(self.y_low_text.text())
        except ValueError:
            print("y low angle is not a number")
            return

        try:
            y_high = float(self.y_high_text.text())
        except ValueError:
            print("y high angle is not a number")
            return

        try:
            z_low = float(self.z_low_text.text())
        except ValueError:
            print("z low angle is not a number")
            return

        try:
            z_high = float(self.z_high_text.text())
        except ValueError:
            print("z high angle is not a number")
            return

        try:
            view_step = float(self.view_step.text())
        except ValueError:
            print("View step is not a number")
            return

        curr_euler = self.euler_order.currentText()
        print(x_low, x_high, y_low, y_high, z_low, z_high, curr_euler, view_step)

        xs: np.ndarray = np.arange(x_low, x_high, view_step)
        ys: np.ndarray = np.arange(y_low, y_high, view_step)
        zs: np.ndarray = np.arange(z_low, z_high, view_step)

        euler_tmp = self.cartesian_product_simple_transpose([xs, ys, zs])
        rot_tmp: Rotation = Rotation.from_euler(curr_euler, euler_tmp, degrees=True)

        return rot_tmp

    @staticmethod
    def draw_3d_points(res: np.ndarray):
        ax = plt.axes(projection='3d')
        ax.scatter3D(res[:, 0], res[:, 1], res[:, 2])
        plt.show()

    def draw_button_func(self):
        rot_tmp = self.parse_euler_tmp()
        res = rot_tmp.apply(np.array([0, 0, 1]))
        self.draw_3d_points(res)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WindowClass()
    win.show()
    sys.exit(app.exec_())
