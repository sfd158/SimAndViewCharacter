import math
import numpy as np
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize, QEvent
from PyQt5.QtGui import QColor, QVector3D, QMatrix4x4, QWheelEvent, QMouseEvent, QNativeGestureEvent
from PyQt5.QtWidgets import QOpenGLWidget, QWidget, QGestureEvent, QPinchGesture, QSwipeGesture, QPanGesture

try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
except ImportError as err:
    print(err)
    raise err

try:
    from . import DrawPrimitives as dp
except ImportError:
    import DrawPrimitives as dp


class ViewWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # camera
        self.cameraPosition = QVector3D(0, 2, -2)
        self.cameraCenter = QVector3D(0, 1, 0)
        self.cameraUp = QVector3D(0, 1, 0)
        self.viewMatrix = QMatrix4x4()
        self.look()

        # for camera control
        self.lastPos = QPoint()
        self.lastCameraPosition = self.cameraPosition
        self.lastCameraCenter = self.cameraCenter
        self.lastCameraUp = QVector3D(0, 1, 0)

        # objects
        self.renderables = []

        self.grid = 0

        # Add by Zhenhua Song
        self.pos3d_render_buf: Optional[np.ndarray] = None

    def getOpenGLInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )

        return info

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        print(self.getOpenGLInfo())
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glShadeModel(gl.GL_FLAT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        self.grid = self.createGrid()

    # Add by Zhenhua Song
    def draw_3d_trajectory(self):
        buf: np.ndarray = self.pos3d_render_buf
        if buf is None or buf.ndim !=2 or buf.shape[-1] != 3:
            return
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        for i in range(buf.shape[0]):
            gl.glVertex3d(*buf[i])
        gl.glEnd()

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # set up view matrix
        gl.glLoadIdentity()
        gl.glLoadMatrixd(self.viewMatrix.data())

        self.draw_3d_trajectory()
        self.draw()

        # draw other objects
        for obj in self.renderables:
            gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
            gl.glPushMatrix()

            obj.draw()

            gl.glPopMatrix()
            gl.glPopAttrib()

    def draw(self):
        # ground grid
        gl.glCallList(self.grid)

        # coordinates
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glPushMatrix()

        gl.glColor3f(0.5, 0.5, 0.5)
        dp.drawSphere(True, 0.019, 10, 10)

        gl.glColor3f(0, 0, 1)
        dp.drawCone(True, 0.02, 1, 10, 10)

        gl.glRotated(-90, 1, 0, 0)
        gl.glColor3f(0, 1, 0)
        dp.drawCone(True, 0.02, 1, 10, 10)

        gl.glRotated(90, 0, 1, 0)
        gl.glColor3f(1, 0, 0)
        dp.drawCone(True, 0.02, 1, 10, 10)

        gl.glPopMatrix()
        gl.glPopAttrib()
        pass

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport(0, 0, width, height)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, width / height, 0.1, 100)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def look(self):
        self.viewMatrix.setToIdentity()
        self.viewMatrix.lookAt(self.cameraPosition, self.cameraCenter, QVector3D(0, 1, 0))
        self.update()

    def mousePressEvent(self, e:QMouseEvent):
        self.lastPos = e.pos()
        self.lastCameraPosition = QVector3D(self.cameraPosition)
        self.lastCameraCenter = QVector3D(self.cameraCenter)

    def mouseReleaseEvent(self, e):
        self.lastPos = e.pos()
        self.lastCameraPosition = QVector3D(self.cameraPosition)
        self.lastCameraCenter = QVector3D(self.cameraCenter)

    def mouseMoveEvent(self, e):
        mousePosOff  = e.pos() - self.lastPos

        sz = self.size()

        # middle button: rise fall shift
        if e.buttons() & Qt.MidButton or \
            (e.buttons() & Qt.LeftButton and e.modifiers() & Qt.ControlModifier):
            z = self.lastCameraPosition - self.lastCameraCenter

            shiftScale = 1.0 * z.length()

            x = QVector3D.crossProduct(self.lastCameraUp, z)
            z.normalize()
            x.normalize()
            y = QVector3D.crossProduct(z, x)

            shift = -(mousePosOff.x() / sz.height()) * x + (mousePosOff.y() / sz.height()) * y
            shift *= shiftScale
            self.cameraPosition = self.lastCameraPosition + shift
            self.cameraCenter = self.lastCameraCenter + shift

            self.look()

        # right button: move back and forth
        elif (e.buttons() & Qt.RightButton) or \
            (e.buttons() & Qt.LeftButton and e.modifiers() & Qt.ShiftModifier):
            z = self.lastCameraPosition - self.lastCameraCenter
            scale = 1.0
            scale = (1.0 + (scale * mousePosOff.y() / sz.height()))
            if (scale < 0.05):
                scale = 0.05
            self.cameraPosition = self.lastCameraCenter + scale * z

            self.look()

        # left button: pan tilt
        elif e.buttons() & Qt.LeftButton:
            z = self.lastCameraPosition - self.lastCameraCenter

            zDotUp = QVector3D.dotProduct(self.lastCameraUp, z)
            zmap = z - zDotUp * self.lastCameraUp
            angx = math.degrees(math.acos(zmap.length() / z.length()))
            if zDotUp < 0:
                angx = -angx

            angleScale = 200.0

            x = QVector3D.crossProduct(self.lastCameraUp, z)
            x.normalize()
            y = QVector3D.crossProduct(z, x)
            y.normalize()

            rotXang = mousePosOff.y() / sz.height() * angleScale
            rotXang += angx
            if (rotXang > 85):
                rotXang = 85
            if (rotXang < -85):
                rotXang = -85
            rotXang -= angx

            rot = QMatrix4x4()
            rot.rotate(-mousePosOff.x() / sz.height() * angleScale, y)
            rot.rotate(-rotXang, x)

            self.cameraPosition = self.cameraCenter + rot * z

            self.look()

    def event(self, event:QEvent):
        if event.type() == 197: #QEvent.NativeGesture
            return self.nativeGuestureEvent(event)

        return super().event(event)

    def nativeGuestureEvent(self, event:QNativeGestureEvent):
        # only for Mac
        if event.gestureType() == Qt.BeginNativeGesture:
            self.lastCameraPosition = QVector3D(self.cameraPosition)
            self.lastCameraCenter = QVector3D(self.cameraCenter)

        elif event.gestureType() == Qt.ZoomNativeGesture:

            z = self.lastCameraPosition - self.lastCameraCenter
            scale = (1.0 - 1 * event.value())
            z *= scale
            l = z.length()
            if l < 0.05:
                z = z / l * 0.05
            self.cameraPosition = self.lastCameraCenter + z
            self.lastCameraPosition = QVector3D(self.cameraPosition)

            self.look()

        elif event.gestureType() == Qt.SmartZoomNativeGesture:
            return False

        elif event.gestureType() == Qt.RotateNativeGesture:
            return False

        elif event.gestureType() == Qt.SwipeNativeGesture:
            return False

        return True

    def createGrid(self):
        genList = gl.glGenLists(1)
        gl.glNewList(genList, gl.GL_COMPILE)

        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.7, 0.7, 0.7)

        for i in range(-10, 11, 1):
            gl.glVertex3d(i, 0,  10)
            gl.glVertex3d(i, 0, -10)
            gl.glVertex3d( 10, 0, i)
            gl.glVertex3d(-10, 0, i)

        gl.glEnd()

        gl.glEndList()

        return genList


def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ViewWidget()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
