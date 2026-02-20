"""
app.py – PyQt6 main window and application entry-point for SAM 3D Body Analyzer.
"""

import os
import sys

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal

from model.sam3d_processor import SAM3DProcessor
from visualization.viewer_3d import Viewer3D


# ── Background processing thread ──────────────────────────────────────────────

class _ProcessingThread(QThread):
    """Runs SAM3DProcessor on the selected video without blocking the UI."""

    log_emitted = pyqtSignal(str)
    frame_emitted = pyqtSignal(object)   # carries np.ndarray (BGR annotated frame)
    cloud_emitted = pyqtSignal(object)   # carries np.ndarray (N×3 point cloud)
    finished_processing = pyqtSignal()

    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self._video_path = video_path

    def run(self) -> None:
        processor = SAM3DProcessor(log_callback=self.log_emitted.emit)

        def on_frame(annotated: np.ndarray, pts: np.ndarray) -> None:
            self.frame_emitted.emit(annotated.copy())
            if pts is not None and len(pts) > 0:
                self.cloud_emitted.emit(pts.copy())

        processor.run(self._video_path, frame_callback=on_frame)
        self.finished_processing.emit()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAM 3D Body Analyzer")
        self.setMinimumSize(900, 720)
        self._viewer = Viewer3D()
        self._thread: _ProcessingThread | None = None
        self._video_path: str = ""
        self._all_points: list[np.ndarray] = []
        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Row 1: file selection ──────────────────────────────────────────
        top_row = QtWidgets.QHBoxLayout()
        self._btn_select = QtWidgets.QPushButton("📂  Select Video…")
        self._btn_select.setFixedWidth(160)
        self._btn_select.clicked.connect(self._select_video)
        top_row.addWidget(self._btn_select)

        self._lbl_path = QtWidgets.QLabel("No video selected.")
        self._lbl_path.setWordWrap(True)
        top_row.addWidget(self._lbl_path, stretch=1)
        root.addLayout(top_row)

        # ── Row 2: video thumbnail ─────────────────────────────────────────
        self._lbl_frame = QtWidgets.QLabel("Video frame will appear here")
        self._lbl_frame.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lbl_frame.setMinimumSize(320, 240)
        self._lbl_frame.setStyleSheet(
            "background:#1a1a2e; border:1px solid #444; color:#888;"
        )
        root.addWidget(self._lbl_frame, stretch=2)

        # ── Row 3: action buttons ──────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_process = QtWidgets.QPushButton("▶  Run SAM 3D Body Processing")
        self._btn_process.setEnabled(False)
        self._btn_process.clicked.connect(self._start_processing)
        btn_row.addWidget(self._btn_process)

        self._btn_show3d = QtWidgets.QPushButton("🧊  Show 3D Scene")
        self._btn_show3d.setEnabled(False)
        self._btn_show3d.clicked.connect(self._show_3d)
        btn_row.addWidget(self._btn_show3d)
        root.addLayout(btn_row)

        # ── Row 4: log output ──────────────────────────────────────────────
        root.addWidget(QtWidgets.QLabel("Model Logs:"))
        self._log_box = QtWidgets.QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setStyleSheet(
            "background:#0d1117; color:#58a6ff; font-family:monospace;"
        )
        self._log_box.setMinimumHeight(160)
        root.addWidget(self._log_box, stretch=1)

        self.statusBar().showMessage("Ready – select a video to begin.")

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _select_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if not path:
            return
        self._video_path = path
        self._lbl_path.setText(path)
        self._btn_process.setEnabled(True)
        self._all_points.clear()
        self._btn_show3d.setEnabled(False)
        self._log_box.clear()
        self._show_first_frame(path)
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")

    def _show_first_frame(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._display_frame(frame)

    def _display_frame(self, bgr_frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(
            rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._lbl_frame.width(),
            self._lbl_frame.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._lbl_frame.setPixmap(scaled)

    def _append_log(self, message: str) -> None:
        self._log_box.append(message)
        sb = self._log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _start_processing(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        self._log_box.clear()
        self._all_points.clear()
        self._btn_process.setEnabled(False)
        self._btn_show3d.setEnabled(False)
        self.statusBar().showMessage("Processing…")

        self._thread = _ProcessingThread(self._video_path, parent=self)
        self._thread.log_emitted.connect(self._append_log)
        self._thread.frame_emitted.connect(self._display_frame)
        self._thread.cloud_emitted.connect(self._accumulate_cloud)
        self._thread.finished_processing.connect(self._on_processing_done)
        self._thread.start()

    def _accumulate_cloud(self, pts: np.ndarray) -> None:
        self._all_points.append(pts)

    def _on_processing_done(self) -> None:
        self._btn_process.setEnabled(True)
        if self._all_points:
            self._btn_show3d.setEnabled(True)
        self.statusBar().showMessage("Processing complete.")
        self._append_log("─── Processing finished ───")

    def _show_3d(self) -> None:
        if not self._all_points:
            QtWidgets.QMessageBox.information(self, "No Data", "No 3D data to display.")
            return
        combined = np.vstack(self._all_points)
        self._viewer.show(combined)


# ── Application wrapper ───────────────────────────────────────────────────────

class SAM3DApp:
    """Thin wrapper that owns the QApplication and the main window."""

    def __init__(self) -> None:
        self._qt_app = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication(sys.argv)
        )
        self._window = MainWindow()

    def run(self) -> None:
        self._window.show()
        sys.exit(self._qt_app.exec())


# ── Stand-alone entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    app = SAM3DApp()
    app.run()
