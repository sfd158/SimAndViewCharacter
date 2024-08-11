import numpy as np
try:
    import OpenGL
except ImportError as err:
    print(err)
    raise err

import OpenGL.GL as gl
import OpenGL.GLU as glu
import os
from PyQt5.QtGui import QQuaternion, QVector3D, QColor
from typing import List, Optional


try:
    from .View import ViewWidget
    from .Timeline import TimelineWidget
    from . import DrawPrimitives as dp
except ImportError:
    from View import ViewWidget
    from Timeline import TimelineWidget
    import DrawPrimitives as dp

try:
    from ..MotionData import MotionData
    from .. import BVHLoader
    from ...Common.MathHelper import MathHelper
except ImportError:
    from VclSimuBackend.pymotionlib.MotionData import MotionData
    from VclSimuBackend.pymotionlib import BVHLoader


class Skeleton:
    def __init__(self, jointNames, jointParents, jointOffsets, jointOrientations, color: Optional[QColor] = None):
        self.jointNames = jointNames
        self.jointParents = jointParents
        self.jointChildren = [[] for _ in jointNames]
        for i, p in enumerate(jointParents[1:]):
            self.jointChildren[p].append(i+1)

        if color is None:
            self.jointColors = [
                (1.0, 0.25, 0.25, 1.0) if 'left' in jn or 'Left' in jn or 'LEFT' in jn or \
                                        (jn[0] == 'l' and 'r'+jn[1:] in jointNames) or \
                                        (jn[0] == 'L' and 'R'+jn[1:] in jointNames) else
                (0.25, 0.25, 1.0, 1.0) if 'right' in jn or 'Right' in jn or 'RIGHT' in jn or \
                                        (jn[0] == 'r' and 'l'+jn[1:] in jointNames) or \
                                        (jn[0] == 'R' and 'L'+jn[1:] in jointNames) else
                (0.25, 0.25, 0.25, 0.25)
                for jn in jointNames
            ]
        else:
            self.jointColors = [(color.red() / 255, color.green() / 255, color.blue() / 255, color.alpha() / 255)
                                for _ in jointNames]

        self.jointOffsets = jointOffsets
        self.jointOrientations = jointOrientations


class ClipFrame:
    def __init__(self, jointTranslations, jointRotations):
        self.jointTranslations = jointTranslations
        self.jointRotations = jointRotations


class Clip:
    def __init__(self, motion: MotionData, frameOffset: int):
        self.motion = motion
        self.initial_motion = motion.sub_sequence(copy=True)
        self.frameOffset = frameOffset

    def load_initial_motion(self):
        if self.initial_motion is not None:
            self.motion = self.initial_motion.sub_sequence(copy=True)
        return self.motion

    def convert_to_facing_coordinate(self):
        if self.motion is not None:
            self.motion = self.motion.to_facing_coordinate()
        return self.motion

    def covert_to_local_coordinate(self):
        if self.motion is not None:
            self.motion = self.motion.to_local_coordinate()
        return self.motion

    def getFrame(self, frame: int):
        if self.motion is None:
            return None

        if frame < self.frameOffset:
            return None

        frame -= self.frameOffset
        if frame >= self.motion.num_frames:
            return None

        return ClipFrame(self.motion.joint_translation[frame], self.motion.joint_rotation[frame])


class RenderAgent:
    def __init__(self, skeleton: Skeleton, clipFrame: ClipFrame = None):
        self.skeleton = skeleton
        self.color = [0.25, 0.25, 0.25, 1.0]
        self.clipFrame = clipFrame

    def draw(self):
        if self.skeleton is None:
            return

        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColor4dv(self.color)
        self._drawBone(0)

        gl.glPopAttrib()

    def _drawBone(self, idx: int):
        gl.glPushMatrix()

        trans = self.skeleton.jointOffsets[idx]
        rot = QQuaternion(*self.skeleton.jointOrientations[idx,(3, 0, 1, 2)])

        if self.clipFrame is not None:
            trans = trans + self.clipFrame.jointTranslations[idx]
            rot = rot * QQuaternion(*self.clipFrame.jointRotations[idx,(3, 0, 1, 2)])

        gl.glTranslated(trans[0], trans[1], trans[2])
        gl.glColor4dv(self.skeleton.jointColors[idx])
        dp.drawSphere(False, 0.015, 6, 3)

        if idx > 0:
            distance = np.linalg.norm(trans)
            if distance > 0.02:
                # bone point
                gl.glPushMatrix()
                axisAngle = QQuaternion.rotationTo(QVector3D(0, 0, 1), QVector3D(*-trans)).getAxisAndAngle()
                gl.glRotated(axisAngle[1], axisAngle[0].x(), axisAngle[0].y(), axisAngle[0].z())
                dp.drawCone(False, 0.015, distance, 6, 3)
                gl.glPopMatrix()

        axisAngle = rot.getAxisAndAngle()
        gl.glRotated(axisAngle[1], axisAngle[0].x(), axisAngle[0].y(), axisAngle[0].z())

        for child in self.skeleton.jointChildren[idx]:
            self._drawBone(child)

        gl.glPopMatrix()



class _ClipInfo:
    def __init__(self) -> None:
        self.skeleton = None
        self.renderAgent = None
        self.clip = None
        self.name = ""
        self.num_frames = 0

    def load_init_motion(self):
        if self.clip is not None:
            self.clip.load_initial_motion()

    def to_facing_coordinate(self):
        if self.clip is not None:
            self.clip.convert_to_facing_coordinate()

    def to_local_coordinate(self):
        if self.clip is not None:
            self.clip.covert_to_local_coordinate()

class ModifiedClipManager:
    def __init__(self, view: ViewWidget, timeline: TimelineWidget):
        self.view = view
        self.timeline = timeline

        self.clip_list: List[_ClipInfo] = []
        self.total_frames = 0

        self.currentFrame: int = 0
        timeline.frameSlider.valueChanged.connect(self.setCurrentFrame)

    # Add by Zhenhua Song
    def load_initial_motion(self):
        if self.clip_list is not None:
            for node in self.clip_list:
                if node is not None:
                    node.load_init_motion()

    # Add by Zhenhua Song
    def to_facing_coordinate(self):
        if self.clip_list is not None:
            for node in self.clip_list:
                if node is not None:
                    node.to_facing_coordinate()

    def to_local_coordinate(self):
        if self.clip_list is not None:
            for node in self.clip_list:
                if node is not None:
                    node.to_local_coordinate()

    def load(self, fn: str, scale: float, color: QColor, new_fps = None):
        assert fn.endswith(".bvh")
        _info = _ClipInfo()
        data: MotionData = BVHLoader.load(fn).scale(scale)
        if new_fps:
            data.resample(new_fps)

        if self.total_frames < data.num_frames:
            self.total_frames = data.num_frames
            self.timeline.setRange(0, data.num_frames)

        _info.skeleton = Skeleton(
                data.joint_names,
                data.joint_parents_idx,
                data.joint_offsets,
                np.tile((0, 0, 0, 1.), (data.num_joints, 1)),
                color
            )
        _info.renderAgent = RenderAgent(_info.skeleton)
        _info.clip = Clip(data, 0)
        _info.num_frames = data.num_frames
        _info.name = os.path.split(fn)[1]

        self.view.renderables.append(_info.renderAgent)
        self.clip_list.append(_info)
        self.update()
    
    def loadSkeleton(self, fn):
        raise NotImplementedError()

    def setCurrentFrame(self, frame: int):
        self.currentFrame = frame
        self.update()

    def update(self):
        if not self.clip_list:
            return

        for _info in self.clip_list:
            frame = _info.clip.getFrame(self.currentFrame) if self.currentFrame < _info.num_frames else None
            _info.renderAgent.clipFrame = frame

        self.view.update()

    def showSkeleton(self):
        if not self.clip_list:
            return
        for _info in self.clip_list:
            _info.renderAgent.clipFrame = None

        self.view.update()


class ClipManager:
    def __init__(self, view: ViewWidget, timeline: TimelineWidget):
        self.view = view
        self.timeline = timeline

        self.skeleton = None
        self.renderAgent = None
        self.tracks = []

        self.currentFrame: int = 0
        timeline.frameSlider.valueChanged.connect(self.setCurrentFrame)

    def load(self, fn: str, scale: float):
        assert fn.endswith(".bvh")
        data = BVHLoader.load(fn).scale(scale)

        if self.skeleton is None:
            self.setupSkeleton(Skeleton(
                data.joint_names,
                data.joint_parents_idx,
                data.joint_offsets,
                np.tile((0, 0, 0, 1.), (data.num_joints, 1))
            ))
        else:
            print("we may need to perform retargeting...")
            self.setupSkeleton(Skeleton(
                data.joint_names,
                data.joint_parents_idx,
                data.joint_offsets,
                np.tile((0, 0, 0, 1.), (data.num_joints, 1))
            ))

        # TODO: add multiple-track support
        self.tracks = [[Clip(data, 0)]]
        self.timeline.setRange(0, data.num_frames)
        self.update()

    def loadSkeleton(self, fn):
        raise NotImplementedError()

    def setupSkeleton(self, skeleton:Skeleton):
        self.skeleton = skeleton

        if self.renderAgent is not None:
            self.view.renderables.remove(self.renderAgent)

        self.renderAgent = RenderAgent(skeleton)
        self.view.renderables.append(self.renderAgent)

    def setCurrentFrame(self, frame: int):
        self.currentFrame = frame
        self.update()

    def update(self):
        if self.renderAgent is None:
            return

        clipFrames = []
        for track in self.tracks:
            for clip in track:
                frame = clip.getFrame(self.currentFrame)
                if frame is not None:
                    clipFrames.append(frame)

        # TODO: merge frames
        self.renderAgent.clipFrame = clipFrames[0] if len(clipFrames) > 0 else None
        self.view.update()

    def showSkeleton(self):
        if self.renderAgent is None:
            return

        self.renderAgent.clipFrame = None
        self.view.update()
