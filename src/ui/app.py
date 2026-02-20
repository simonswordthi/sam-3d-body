import sys
from PyQt6 import QtCore, QtWidgets, QtGui


class SAM3DApp(QtWidgets.QMainWindow):
    """Main application window for SAM 3D Body."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 3D Body")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Image / Video input area
        self.video_player = QtWidgets.QLabel("Image / Video Placeholder")
        self.video_player.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_player)

        # Log output
        self.logs_display = QtWidgets.QTextEdit()
        self.logs_display.setReadOnly(True)
        self.logs_display.setPlaceholderText("Logs will be displayed here...")
        layout.addWidget(self.logs_display)

        # 3D viewer placeholder
        self.threed_viewer = QtWidgets.QLabel("3D Viewer Placeholder")
        self.threed_viewer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.threed_viewer)

        self.statusBar().showMessage("Ready")

    def run(self):
        """Show the window and start the Qt event loop."""
        self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SAM3DApp()
    window.run()
    sys.exit(app.exec())
