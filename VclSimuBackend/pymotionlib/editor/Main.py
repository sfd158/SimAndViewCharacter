from argparse import ArgumentParser
from typing import Tuple
import numpy as np
import os
import random
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QKeySequence, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QMainWindow, QApplication, QVBoxLayout, QWidget,
    QAction, QToolBar,
    QMessageBox, QFileDialog, QInputDialog, QColorDialog, QTableWidget, QTableWidgetItem
)
from scipy.spatial.transform import Rotation


try:
    from .View import ViewWidget
    from .ClipMgr import ModifiedClipManager
    from .Timeline import TimelineWidget
    from ..MotionData import MotionData
except ImportError:
    from View import ViewWidget
    from ClipMgr import ModifiedClipManager
    from Timeline import TimelineWidget
    from VclSimuBackend.pymotionlib.MotionData import MotionData


class ColorVisWindow(QWidget):
    def __init__(self, info_list=None):
        super().__init__()
        self.setWindowTitle("Color Vis")
        self.info_list = info_list
        layout = QVBoxLayout()
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(self.info_list))
        heads = ["name", "num frame", "num joint", "color"]
        self.tableWidget.setColumnCount(len(heads))
        self.tableWidget.setHorizontalHeaderLabels(heads)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)
        self.resize(800, 400)

        for i in range(len(info_list)):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(info_list[i].name))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(info_list[i].num_frames)))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(info_list[i].clip.motion.num_joints)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(""))
            color = info_list[i].skeleton.jointColors[0]
            color = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            self.tableWidget.item(i, 3).setBackground(QBrush(color))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.timeline = TimelineWidget(self)
        dock = QToolBar()
        dock.addWidget(self.timeline)
        self.addToolBar(Qt.BottomToolBarArea, dock)

        self.view_widget = ViewWidget()
        self.setCentralWidget(self.view_widget)
        self.clipMgr = ModifiedClipManager(self.view_widget, self.timeline)
        self.color_win = None
        self.createActions()
        self.createMenus()

        message = "A context menu is available by right-clicking"
        self.statusBar().showMessage(message)

        self.setWindowTitle("Menus")
        self.setMinimumSize(160, 160)
        self.resize(800, 600)

        parser = ArgumentParser()
        parser.add_argument("--bvh_fname", nargs="*")
        args = parser.parse_args()
        if args.bvh_fname:
            if args.bvh_fname[0] == "*":
                args.bvh_fname = [fname for fname in os.listdir(os.getcwd()) if fname.endswith(".bvh")]
            for fname in args.bvh_fname:
                if os.path.isfile(fname):
                    self._open_hlp(fname)

    # def contextMenuEvent(self, event):  # remove by Zhenhua Song
    #     menu = QMenu(self)
    #     action_t_pose = QAction('Show &T-pose')
    #     action_t_pose.triggered.connect(lambda: self.clipMgr.showSkeleton() )
    #     menu.addAction(action_t_pose)

    #     menu.exec_(event.globalPos())

    def closeEvent(self, event):
        sys.exit(0)

    def newFile(self):
        self.statusBar().showMessage("Invoked <b>File|New</b>")

    def _open_hlp(self, fn: str, fps = None):
        col = QColor(*[random.randint(0, 255) for _ in range(3)])
        print(col.name())
        self.clipMgr.load(fn, 1.0, col, fps)

    def open(self):
        fn = QFileDialog.getOpenFileName(self, 'Load Motion File',
            filter='Motion files (*.bvh);;All files (*.*)')

        if fn[0] == '':
            return

        # Modify by Zhenhua Song
        # scale = QInputDialog.getDouble(self, 'Scale: ', 'Data Scale', 1, decimals=6)
        # print(scale)

        # if scale[1]:
        #     scale = scale[0]
        # else:
        #     scale = 1.0
        # col = QColorDialog.getColor()
        self._open_hlp(fn[0])

    def openResample(self):
        fn = QFileDialog.getOpenFileName(self, 'Load Motion File',
            filter='Motion files (*.bvh);;All files (*.*)')
        if fn[0] == '':
            return
        self._open_hlp(fn[0], 60)
    
    def save(self):
        self.statusBar().showMessage("Invoked <b>File|Save</b>")

    def vis_color(self):
        self.statusBar().showMessage("Invoked <b>Vis Color</b>")
        self.color_win = ColorVisWindow(self.clipMgr.clip_list)
        self.color_win.show()

    def export_3d_func(self):
        self.statusBar().showMessage("Invoked Export 3d Func")

    def export_2d_func(self):
        self.statusBar().showMessage("Invoked Export 2d Func")

    def show_camera_pos_func(self):
        self.statusBar().showMessage("Invoked Show Camera Pos Func")
        pos = self.view_widget.cameraPosition
        print(f"Camera Pos = [{pos.x()}, {pos.y()}, {pos.z()}]")

    def show_camera_center_func(self):
        self.statusBar().showMessage("Invoked Show Camera Center Func")
        center = self.view_widget.cameraCenter
        print(f"Camera Center = [{center.x()}, {center.y()}, {center.z()}]")

    def show_camera_up_func(self):
        self.statusBar().showMessage("Invoked Show Camera Up Func")
        up = self.view_widget.cameraUp
        print(f"Camera Up = [{up.x()}, {up.y()}, {up.z()}]")

    def undo(self):
        self.statusBar().showMessage("Invoked <b>Edit|Undo</b>")

    def redo(self):
        self.statusBar().showMessage("Invoked <b>Edit|Redo</b>")

    def cut(self):
        self.statusBar().showMessage("Invoked <b>Edit|Cut</b>")

    def copy(self):
        self.statusBar().showMessage("Invoked <b>Edit|Copy</b>")

    def paste(self):
        self.statusBar().showMessage("Invoked <b>Edit|Paste</b>")

    def reset(self):
        self.view_widget = ViewWidget()
        self.setCentralWidget(self.view_widget)
        self.clipMgr = ModifiedClipManager(self.view_widget, self.timeline)
        self.color_win = None
    
    @staticmethod
    def print_motion_range(motion: MotionData):
        x_max, x_min = np.max(motion.joint_position[..., 0]), np.min(motion.joint_position[..., 0])
        y_max, y_min = np.max(motion.joint_position[..., 1]), np.min(motion.joint_position[..., 1])
        z_max, z_min = np.max(motion.joint_position[..., 2]), np.min(motion.joint_position[..., 2])
        print(f"x_min = {x_min:.3f}, x_max = {x_max:.3f}")
        print(f"y_min = {y_min:.3f}, y_max = {y_max:.3f}")
        print(f"z_min = {z_min:.3f}, z_max = {z_max:.3f}")
        print()

    def print_joint_global_pos(self):
        motion: MotionData = self.select_motion(0)
        if motion is None:
            return

        frame = self.timeline.frameSlider.value()
        for joint_idx in range(motion.num_joints):
            if motion.end_sites is not None and joint_idx in motion.end_sites:
                continue
            info = " ".join([f"{motion.joint_position[frame, joint_idx, i]:.3f}" for i in range(3)])
            print(f"{motion.joint_names[joint_idx]}, {info}")
        print()

    def enable_render_joint_pos(self):
        self.statusBar().showMessage("Invoked render joint pos")
        motion: MotionData = self.select_motion(0)
        choice_names = [f"{idx}.{name}" for idx, name in enumerate(motion.joint_names)] + ["NoRender"]
        choice_dict = {name: idx for idx, name in enumerate(choice_names)}
        choice_dict["NoRender"] = -1
        res, success = QInputDialog.getItem(self, "RenderJoint", "select joint", choice_names)
        if not success:
            return
        joint_index: int = choice_dict.get(res, None)
        if joint_index is None or joint_index == -1:
            return
        self.view_widget.pos3d_render_buf = motion.joint_position[:, joint_index, :]

    def vis_in_facing(self):
        self.statusBar().showMessage("Visualize in facing coordinate")
        self.clipMgr.load_initial_motion()
        self.clipMgr.to_facing_coordinate()

    def vis_in_local(self):
        self.statusBar().showMessage("Visualize in local coordinate")
        self.clipMgr.load_initial_motion()
        self.clipMgr.to_local_coordinate()

    def select_motion(self, select_index: int):
        if not self.clipMgr.clip_list:
            return None
        motion: MotionData = self.clipMgr.clip_list[select_index].clip.motion
        return motion

    def load_initial_motion(self):
        self.statusBar().showMessage("Load initial motion")
        self.clipMgr.load_initial_motion()

    def y_up_func(self):
        self.statusBar().showMessage("Invoked y_up_func")
        if len(self.clipMgr.clip_list) == 0:
            return
        select_index = 0
        motion = self.select_motion(select_index)
        # rotate as y-up
        # judge if y axis is up

        self.print_motion_range(motion)
        motion.z_up_to_y_up()
        self.print_motion_range(motion)

    def scale_motion_func(self):
        self.statusBar().showMessage("Invoked scale motion func")
        select_index = 0
        motion: MotionData = self.select_motion(select_index)
        scale: Tuple[float, bool] = QInputDialog.getDouble(self, 'Scale: ', 'Data Scale', 1, decimals=6)
        if scale[1]:
            motion.scale(scale[0])
        self.print_motion_range(motion)

    def about(self):
        self.statusBar().showMessage("Invoked <b>Help|About</b>")
        QMessageBox.about(self, "About Menu",
                "The <b>Menu</b> example shows how to create menu-bar menus "
                "and context menus.")

    def aboutQt(self):
        self.statusBar().showMessage("Invoked <b>Help|About Qt</b>")

    def createActions(self):
        self.newAct = QAction("&New", self, shortcut=QKeySequence.New,
                statusTip="Create a new file", triggered=self.newFile)

        self.openAct = QAction("&Open...", self, shortcut=QKeySequence.Open,
                statusTip="Open an existing file", triggered=self.open)
        
        # Added by Heyuan Yao
        self.openResampleAct = QAction("&OpenResample", self, shortcut="Alt+O",
                statusTip="Open an existing file and Resample to 60", triggered=self.openResample)
        self.resetAct = QAction("&Reset", self, shortcut = "Ctrl+R",
                statusTip="clear all opened bvh", triggered=self.reset)
        
        self.saveAct = QAction("&Save", self, shortcut=QKeySequence.Save,
                statusTip="Save the document to disk", triggered=self.save)

        self.visColorAct = QAction("&Vis Color", self,shortcut="Ctrl+I", triggered=self.vis_color)

        self.exitAct = QAction(" E&xit", self, shortcut="Ctrl+Q",
                statusTip="Exit the application", triggered=self.close)

        self.cameraPosAct = QAction("Camera Pos", self, statusTip="Camera Position", triggered=self.show_camera_pos_func)
        self.cameraCenterAct = QAction("Camera Center", self, statusTip="Camera Center", triggered=self.show_camera_center_func)
        self.cameraUpAct = QAction("Camera Up", self, statusTip="Camera Up", triggered=self.show_camera_up_func)

        self.export3dAct = QAction("Export 3d", self, statusTip="Export 3d motion to video", triggered=self.export_3d_func)
        self.export2dAct = QAction("Export 2d", self, statusTip="Export 2d motion to video", triggered=self.export_2d_func)

        self.undoAct = QAction("&Undo", self, shortcut=QKeySequence.Undo,
                statusTip="Undo the last operation", triggered=self.undo)

        self.redoAct = QAction("&Redo", self, shortcut=QKeySequence.Redo,
                statusTip="Redo the last operation", triggered=self.redo)

        self.cutAct = QAction("Cu&t", self, shortcut=QKeySequence.Cut,
                statusTip="Cut the current selection's contents to the clipboard",
                triggered=self.cut)

        self.copyAct = QAction("&Copy", self, shortcut=QKeySequence.Copy,
                statusTip="Copy the current selection's contents to the clipboard",
                triggered=self.copy)

        self.pasteAct = QAction("&Paste", self, shortcut=QKeySequence.Paste,
                statusTip="Paste the clipboard's contents into the current selection",
                triggered=self.paste)

        self.yUpAct = QAction("Y Up", self, statusTip="Rotate motion as Y up", triggered=self.y_up_func)

        self.motionScaleAct = QAction("Scale", self, statusTip="Scale Motion", triggered=self.scale_motion_func)

        self.printGlobalAct = QAction("GlobalPos", self, statusTip="Show global pos", triggered=self.print_joint_global_pos)

        self.jointTrajectoryAct = QAction("Joint Trajectory", self, statusTip="Render Joint Trajectory", triggered=self.enable_render_joint_pos)

        self.visInFacingAct = QAction("Facing Coordinate", self, statusTip="Facing Coordinate", triggered=self.vis_in_facing)

        self.visInLocalAct = QAction("Local Coordinate", self, statusTip="Local Coordinate", triggered=self.vis_in_local)

        self.loadInitialAct = QAction("Initial Action", self, statusTip="Initial Motion", triggered=self.load_initial_motion)

        self.aboutAct = QAction("&About", self,
                statusTip="Show the application's About box",
                triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                statusTip="Show the Qt library's About box",
                triggered=self.aboutQt)
        self.aboutQtAct.triggered.connect(QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.newAct)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.saveAct)
        self.fileMenu.addAction(self.visColorAct)
        self.fileMenu.addAction(self.openResampleAct)
        self.fileMenu.addAction(self.resetAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.undoAct.setDisabled(True)
        self.redoAct.setDisabled(True)
        self.cutAct.setDisabled(True)
        self.copyAct.setDisabled(True)
        self.pasteAct.setDisabled(True)

        self.editMenu = self.menuBar().addMenu("&Edit")
        self.editMenu.addAction(self.cameraPosAct)
        self.editMenu.addAction(self.cameraCenterAct)
        self.editMenu.addAction(self.cameraUpAct)

        self.editMenu.addSeparator()
        self.editMenu.addAction(self.export3dAct)
        self.editMenu.addAction(self.export2dAct)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.undoAct)
        self.editMenu.addAction(self.redoAct)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.cutAct)
        self.editMenu.addAction(self.copyAct)
        self.editMenu.addAction(self.pasteAct)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.yUpAct)
        self.editMenu.addAction(self.motionScaleAct)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.printGlobalAct)
        self.editMenu.addAction(self.jointTrajectoryAct)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.visInFacingAct)
        self.editMenu.addAction(self.visInLocalAct)
        self.editMenu.addAction(self.loadInitialAct)
        # self.editMenu.setDisabled(True)

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    app = QApplication(argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
