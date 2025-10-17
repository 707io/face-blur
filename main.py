import sys
import cv2
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QGroupBox, QMessageBox, QComboBox, QSlider
)

CASCADE_FILENAME = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

class FaceBlurGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Face Blur - by 707 (github.com/707io/)")
        self.setMinimumSize(QSize(900, 650))
        self.cap = None
        self.mode = "blur"
        self.pixel_size = 10
        self.dark_mode = True
        self.last_frame = None

        self.faceCascade = cv2.CascadeClassifier(CASCADE_FILENAME)
        if self.faceCascade.empty():
            QMessageBox.critical(self, "Cascade Error",
                                 f"Failed to load cascade file:\n{CASCADE_FILENAME}")
            raise FileNotFoundError(f"Failed to load cascade: {CASCADE_FILENAME}")

        self.video_label = QLabel("Camera not started")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("Status: Idle")

        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setEnabled(False)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Blur", "Pixelate"])

        self.pixel_slider = QSlider(Qt.Horizontal)
        self.pixel_slider.setRange(2, 50)
        self.pixel_slider.setValue(self.pixel_size)
        self.pixel_label = QLabel(f"Pixel Size: {self.pixel_size}")

        self.theme_button = QPushButton("Toggle Theme")
        self.pixel_slider.valueChanged.connect(self.update_pixel_size)

        controls = QGroupBox("Controls")
        h = QHBoxLayout()
        h.addWidget(self.btn_start)
        h.addWidget(self.btn_stop)
        h.addWidget(self.mode_combo)
        h.addWidget(self.pixel_label)
        h.addWidget(self.pixel_slider)
        h.addWidget(self.theme_button)
        controls.setLayout(h)

        v = QVBoxLayout()
        v.addWidget(self.video_label, stretch=1)
        v.addWidget(controls)
        v.addWidget(self.status_label)
        self.setLayout(v)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.fps = 20

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        self.theme_button.clicked.connect(self.toggle_theme)

        self.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #121212; color: #E0E0E0; }
                QPushButton { background-color: #1E1E1E; border: 1px solid #444; padding: 5px; }
                QPushButton:hover { background-color: #333; }
                QComboBox, QSlider::groove:horizontal { background-color: #2E2E2E; }
                QSlider::handle:horizontal { background-color: #555; width: 14px; }
                QLabel { color: #E0E0E0; }
            """)
            self.video_label.setStyleSheet("background-color: #000; color: #fff;")
        else:
            self.setStyleSheet("""
                QWidget { background-color: #FAFAFA; color: #202020; }
                QPushButton { background-color: #E0E0E0; border: 1px solid #AAA; padding: 5px; }
                QPushButton:hover { background-color: #DDD; }
                QComboBox, QSlider::groove:horizontal { background-color: #DDD; }
                QSlider::handle:horizontal { background-color: #888; width: 14px; }
                QLabel { color: #202020; }
            """)
            self.video_label.setStyleSheet("background-color: #EEE; color: #000;")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def update_pixel_size(self, value):
        self.pixel_size = value
        self.pixel_label.setText(f"Pixel Size: {value}")

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Unable to open camera.")
                self.cap = None
                return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Status: Camera started")
        self.timer.start(int(1000 / self.fps))

    def stop_camera(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.video_label.setText("Camera stopped")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Status: Stopped")

    def change_mode(self, idx):
        self.mode = "blur" if idx == 0 else "pixelate"

    def update_frame(self):
        if not self.cap:
            return
        ret, img = self.cap.read()
        if not ret or img is None:
            self.status_label.setText("Status: Frame read failed")
            return
        faces = self.faceCascade.detectMultiScale(img, 1.2, 4)
        for (x, y, w, h) in faces:
            ROI = img[y:y+h, x:x+w]
            if self.mode == "blur":
                blur = cv2.GaussianBlur(ROI, (91, 91), 0)
                img[y:y+h, x:x+w] = blur
            else:
                if ROI.size != 0:
                    block = max(1, min(self.pixel_size, w // 2))
                    small = cv2.resize(ROI, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
                    pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    img[y:y+h, x:x+w] = pixel
        if len(faces) == 0:
            cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            self.status_label.setText("Status: No faces found")
        else:
            self.status_label.setText(f"Status: {len(faces)} face(s)")

        self.last_frame = img.copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, e):
        try:
            self.stop_camera()
        except:
            pass
        e.accept()

def main():
    app = QApplication(sys.argv)
    window = FaceBlurGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
