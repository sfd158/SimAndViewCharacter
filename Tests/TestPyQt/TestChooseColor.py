from PyQt5.QtWidgets import QApplication, QColorDialog , QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import sys


class ColorDialog(QWidget):
    def __init__(self ):
        super().__init__()
        self.widget = QWidget(self)
        self.widget.setGeometry(130, 22, 200, 100)
        col = QColorDialog.getColor()
        print(col.name())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qb = ColorDialog()
    qb.show()
    sys.exit(app.exec_())
