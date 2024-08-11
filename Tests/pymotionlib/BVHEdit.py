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
import copy
import math
import numpy as np
import sys
import os
from typing import Optional, List
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QKeySequence, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QAction, QToolBar, QScrollBar, QSlider, QLabel, QScrollArea, QCheckBox, QLineEdit,
    QMessageBox, QFileDialog, QInputDialog, QColorDialog, QTableWidget, QTableWidgetItem
)
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter, BodyInfoState
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader, ODEScene, ODECharacter, JsonCharacterLoader
from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.Common.SmoothOperator import GaussianBase, smooth_operator

fdir = os.path.abspath(os.path.dirname(__file__))
scene_fname = os.path.join(fdir, "../CharacterData/StdHuman/world.json")

class CommonAttrs:
    modify_window: int = 1
    modify_mode: str = "future" # "gaussian"


class HeightViewNode(QWidget):
    max_slider_count = 1000
    max_height = 1.5
    def __init__(self, index: int, curr_state: BodyInfoState, ode_scene: ODEScene, slider_callback, view_nodes: List) -> None:
        super().__init__()
        self.motion_index = index
        self.curr_state = curr_state
        self.ode_scene = ode_scene
        self.ode_character = ode_scene.character0
        self.slider_callback = slider_callback
        self.view_nodes = view_nodes

        self.should_run_callback = True

        layout = QHBoxLayout()
        self.fix_label = QLabel(f"{index}:")
        self.fix_label.setFixedWidth(30)
        layout.addWidget(self.fix_label)

        self.slider = QSlider(Qt.Horizontal)
        min_index = 0
        self.slider.setMinimum(min_index)
        self.slider.setMaximum(self.max_slider_count)
        self.slider.valueChanged.connect(self.value_change)
        layout.addWidget(self.slider)

        self.label = QLabel("0")
        layout.addWidget(self.label)

        self.select_button = QPushButton("select")
        self.select_button.setMaximumWidth(60)
        layout.addWidget(self.select_button)
        self.select_button.pressed.connect(lambda: self.slider_callback(index))

        self.on_ground_label = QLabel("")
        self.on_ground_label.setFixedWidth(10)
        layout.addWidget(self.on_ground_label)

        self.move_button = QPushButton("move")
        self.move_button.setFixedWidth(35)
        self.move_button.pressed.connect(self.move_button_callback)
        layout.addWidget(self.move_button)

        self.select_box = QCheckBox()
        layout.addWidget(self.select_box)

        self.setLayout(layout)

        self.set_bar_value()
        self.change_by_delta_height(0.0)
        # self.judge_contact()

    def judge_contact(self):
        heights = np.ascontiguousarray(self.curr_state.pos.reshape(-1, 3)[:, 1])
        min_index = np.argmin(heights)
        min_velo = self.curr_state.linear_vel.reshape(-1, 3)[min_index]
        print(min_velo)

    def set_bar_value(self):
        self.slider.setValue(int(self.curr_state.pos[1] / self.max_height * self.max_slider_count))

    def get_min_y(self):
        return self.ode_character.get_aabb().reshape(3, 2)[1, 0]

    def change_by_delta_height(self, dh: float):
        pos = self.curr_state.pos.reshape(-1, 3)
        pos[:, 1] += dh
        if self.should_run_callback:
            self.ode_character.load(self.curr_state)
        if self.get_min_y() < 0:
            self.on_ground_label.setText("x")
        else:
            self.on_ground_label.setText("")
        self.label.setText(f"{pos[0, 1]:.3f}")

    def value_change(self):
        if not self.should_run_callback:
            return
        # self.should_run_callback = False
        curr_value = self.slider.value()
        curr_height = curr_value / self.max_slider_count * self.max_height
        dh = curr_height - self.curr_state.pos[1]
        if CommonAttrs.modify_window == 1:
            self.change_by_delta_height(dh)
        else: # compute dh using Gaussian Filter..
            win = CommonAttrs.modify_window
            center_idx: int = math.floor(win // 2)
            min_index = max(0, self.motion_index - center_idx)
            max_index = min(self.motion_index - center_idx + win, len(self.view_nodes))
            characters = self.ode_scene.characters
            if CommonAttrs.modify_mode == "gaussian":
                delta_arr: np.ndarray = np.zeros(win)
                delta_arr[center_idx] = dh
                delta_arr = smooth_operator(delta_arr, GaussianBase(win))
            elif CommonAttrs.modify_mode == "future":
                delta_arr = np.full(win, dh)
            else:
                raise NotImplementedError

            for i in range(min_index, max_index):
                node = self.view_nodes[i]
                node.should_run_callback = False
                node.change_by_delta_height(delta_arr[i - self.motion_index + center_idx])
                node.set_bar_value()
                node.should_run_callback = True
                characters[i - min_index + 1].load(self.view_nodes[i].curr_state)
            self.ode_character.load(self.curr_state)
                
    
    def move_button_callback(self):  # TODO: update bar
        delta_h = -self.get_min_y() + 1e-8
        self.change_by_delta_height(delta_h)
        self.set_bar_value()
        # self.ode_character.move_character_by_delta(np.array([0.0, delta_h, 0.0]))


class ChangeWindow(QWidget):

    def __init__(self, bvh_fname: str, ode_scene: ODEScene):
        super().__init__()
        # self.setWindowTitle("Edit Motion Height")
        
        self.motion = BVHLoader.load(bvh_fname).resample(20)
        self.num_frames = self.motion.num_frames
        bvh2target = BVHToTargetBase(bvh_fname, 20, ode_scene.character0)
        target = bvh2target.init_target()
        set_target = SetTargetToCharacter(ode_scene.character0, target)
        self.state_list = []
        self.original_h: np.ndarray = set_target.target.root.pos[:, 1].copy()
        self.ode_scene = ode_scene
        self.ode_character = ode_scene.character0
        for i in range(self.num_frames):
            set_target.set_character_byframe(i)
            self.state_list.append(self.ode_character.save())

        self.curr_slider = QScrollBar(Qt.Vertical)
        self.curr_slider.setMinimum(0)
        self.curr_slider.setMaximum(self.num_frames)
        self.curr_index = QLabel("0")
        self.curr_index.setFixedWidth(30)
        self.curr_slider.valueChanged.connect(self.slider_callback)

        self.modify_window_text = QLineEdit(f"{CommonAttrs.modify_window}")
        self.modify_window_text.setFixedWidth(30)

        self.modify_window_apply = QPushButton("apply")
        self.modify_window_apply.setFixedWidth(50)
        self.modify_window_apply.pressed.connect(self.apply_window_callback)

        self.prev_button = QPushButton("<<-")
        self.prev_button.setFixedWidth(50)
        self.prev_button.pressed.connect(self.prev_button_callback)

        self.next_button = QPushButton("->>")
        self.next_button.setFixedWidth(50)
        self.next_button.pressed.connect(self.next_button_callback)

        layout_bars = QVBoxLayout()
        self.view_nodes = []
        for i in range(self.num_frames):
            node = HeightViewNode(i, copy.deepcopy(self.state_list[i]), self.ode_scene, self.slider_callback, self.view_nodes)
            self.view_nodes.append(node)
            layout_bars.addWidget(node)
        widget_bars = QWidget()
        widget_bars.setLayout(layout_bars)

        scroll = QScrollArea()
        scroll.setWidget(widget_bars)

        layout_with_slider = QHBoxLayout()
        layout_with_slider.addWidget(scroll)
        layout_with_slider.addWidget(self.curr_slider)
        widget_with_slider = QWidget()
        widget_with_slider.setLayout(layout_with_slider)

        control_layout = QHBoxLayout()
        control_layout.setAlignment(Qt.AlignLeft)
        control_layout.addWidget(self.curr_index)
        control_layout.addWidget(self.modify_window_text)
        control_layout.addWidget(self.modify_window_apply)
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.next_button)
        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        total_layout = QVBoxLayout()
        total_layout.addWidget(control_widget)
        total_layout.addWidget(widget_with_slider)

        self.setLayout(total_layout)

        self.curr_slider.setValue(0)
    
    def prev_button_callback(self):
        prev_index = max(-CommonAttrs.modify_window + self.curr_slider.value(), 0)
        self.curr_slider.setValue(prev_index)

    def next_button_callback(self):
        next_index = min(CommonAttrs.modify_window + self.curr_slider.value(), self.num_frames)
        self.curr_slider.setValue(next_index)

    def slider_callback(self):
        curr_index = self.curr_slider.value()
        self.curr_index.setText(f"{curr_index}")
        if len(self.view_nodes) > curr_index:
            self.ode_character.load(self.view_nodes[curr_index].curr_state)
            if CommonAttrs.modify_window > 1:
                self.apply_window_callback()
    
    def apply_window_callback(self):
        curr_text = self.modify_window_text.text()
        if curr_text.isdigit():
            CommonAttrs.modify_window = int(curr_text)
            for i in range(1, CommonAttrs.modify_window + 1):
                self.ode_scene.characters[i].move_character(np.array([0.0, -2.0, 0.0]))
            index = self.curr_slider.value()
            self.view_nodes[index].value_change()
        else:
            self.modify_window_text.setText(str(CommonAttrs.modify_window))
    

class ViewMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ode_scene = JsonSceneLoader().load_from_file(scene_fname)
        self.ode_viewer = RenderWorld(self.ode_scene)
        for i in range(60):
            loader = JsonCharacterLoader(self.ode_scene.world, self.ode_scene.space)
            ref_character = loader.load(self.ode_scene.character0.config_dict)
            ref_character.is_enable = False
            ref_character.set_render_color(np.array([0.7, 0.7, 0.7]))
            ref_character.move_character(np.array([0.0, -2.0, 0.0]))
            self.ode_scene.characters.append(ref_character)

        self.ode_viewer.start()
        self.ode_viewer.draw_background(1)
        self.ode_viewer.look_at([3,3,3], [0,1,0], [0,1,0])
        self.ode_viewer.track_body(self.ode_scene.character0.bodies[0], False)

        # self.load_action = QAction("&Load", self, triggered=self.load_motion_callback)
        self.save_action = QAction("&Save", self, triggered=self.save_motion_callback)
        self.set_color_action = QAction("Color", self)
        self.file_menu = self.menuBar().addMenu("&File")
        # self.file_menu.addAction(self.load_action)
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.set_color_action)

        self.change_win: Optional[ChangeWindow] = None
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)
        self.setFixedWidth(600)
        self.setFixedHeight(900)

        self.load_motion_callback()

    def load_motion_callback(self):
        fn = QFileDialog.getOpenFileName(self, 'Load Motion File',
            filter='Motion files (*.bvh);;All files (*.*)')[0]
        if not fn:
            return
        print(f"Load motion from {fn}")
        self.setWindowTitle(fn)
        self.change_win = ChangeWindow(fn, self.ode_scene)
        self.setCentralWidget(self.change_win)

    def save_motion_callback(self):
        fn = QFileDialog.getSaveFileName(self, "Save Motion File", filter='Motion files (*.bvh);;All files (*.*)')[0]
        if not fn:
            return
        # modify the height.
        motion = copy.deepcopy(self.change_win.motion)
        heights = np.array([node.curr_state.pos[1] for node in self.change_win.view_nodes])
        motion.joint_translation[:, 0, 1] = heights
        BVHLoader.save(motion, fn)
        print(f"Save motion to {fn}")


def main(argv=None):
    if argv is None:
        argv = sys.argv

    app = QApplication(argv)
    mainWin = ViewMainWindow()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()