''' a timeline UI'''
import platform

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QSlider, QSizePolicy, QPushButton


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        btnLayout = self.createButtons()
        layout.addLayout(btnLayout)

        self.frameEdit = QLineEdit()
        self.frameEdit.setMaximumWidth(30)
        self.frameEdit.setValidator(QIntValidator())
        self.frameSlider = QSlider(Qt.Horizontal)

        def updateEdit(sliderVal):
            self.frameEdit.setText(str(sliderVal))

        def updateSlider(editVal):
            if editVal == '':
                self.frameSlider.setValue(0)
            else:
                self.frameSlider.setValue(int(editVal))

        self.frameSlider.valueChanged.connect(updateEdit)
        self.frameEdit.textEdited.connect(updateSlider)

        layout.addWidget(self.frameSlider)
        layout.addWidget(self.frameEdit)

        layout.setSpacing(3)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self._is_playing = False

    def createButtons(self):
        self.stepBackBtn = QPushButton('<')
        self.stepForwardBtn = QPushButton('>')
        self.pauseBtn = QPushButton('||')
        self.playBtn = QPushButton(':>')
        self.backwardPlayBtn = QPushButton('<:')

        # self.stepBackBtn.setShortcut('Alt + N')
        # self.stepForwardBtn.setShortcut('Alt + M')

        self.stepBackBtn.clicked.connect(lambda : self.step(-1))
        self.stepForwardBtn.clicked.connect(lambda : self.step(1))
        self.playBtn.clicked.connect(lambda : self.play(1))
        self.backwardPlayBtn.clicked.connect(lambda : self.play(-1))
        self.pauseBtn.clicked.connect(self.pause)

        layout = QHBoxLayout()
        layout.setSpacing(3)
        layout.setContentsMargins(0,0,0,0)
        for btn in [self.stepBackBtn,
                    self.backwardPlayBtn,
                    self.pauseBtn,
                    self.playBtn,
                    self.stepForwardBtn,
                    ]:
            btn.setFixedSize(QSize(40,40))
            layout.addWidget(btn)

        if platform.system() == 'Windows':
            self.pauseBtn.setFixedSize(QSize(83, 40))
        else:
            self.pauseBtn.setFixedSize(QSize(71, 40))
        self.pauseBtn.hide()

        self.playOffset = 0
        self.playTimer = QTimer()
        self.playTimer.timeout.connect(lambda: self.step(self.playOffset))

        return layout

    def step(self, offset):
        v = self.frameSlider.value() + offset
        if v > self.frameSlider.maximum():
            v = self.frameSlider.minimum()

        if v < self.frameSlider.minimum():
            v = self.frameSlider.maximum()

        self.frameSlider.setValue(v)

    def play(self, offset):
        self.pauseBtn.show()
        self.playBtn.hide()
        self.backwardPlayBtn.hide()

        self.playOffset = offset
        self.playTimer.start(16)
        self._is_playing = True

    def pause(self):
        self.pauseBtn.hide()
        self.playBtn.show()
        self.backwardPlayBtn.show()

        self.playTimer.stop()
        self._is_playing = False

    def setRange(self, lower, upper):
        self.frameSlider.setRange(lower, upper)

    # Add by Zhenhua Song
    def keyPressEvent(self, event):
        key = event.key()
        # print(key)
        if not self._is_playing:
            if key == Qt.Key.Key_A:
                self.step(-1)
            elif key == Qt.Key.Key_S:
                self.step(1)
