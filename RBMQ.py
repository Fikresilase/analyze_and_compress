import sys
import os
import numpy as np
from PIL import Image, ImageStat
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QVBoxLayout, QWidget, QMessageBox, QTextEdit, QScrollArea
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


def extract_image_properties(img_path):
    try:
        img = Image.open(img_path)
        stats = ImageStat.Stat(img)
        arr = np.array(img.convert("RGB"))

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = np.mean(arr, axis=2)

        properties = []

        # Basic Info
        properties.append(f"Name: {os.path.basename(img_path)}")
        properties.append(f"Format: {img.format}")
        properties.append(f"Mode: {img.mode}")
        properties.append(f"Size (WxH): {img.size}")
        properties.append(f"File Size: {round(os.path.getsize(img_path)/1024, 2)} KB")

        # RGB Stats
        properties += [
            f"Mean R: {np.mean(r):.2f}",
            f"Mean G: {np.mean(g):.2f}",
            f"Mean B: {np.mean(b):.2f}",
            f"Std Dev R: {np.std(r):.2f}",
            f"Std Dev G: {np.std(g):.2f}",
            f"Std Dev B: {np.std(b):.2f}",
            f"Min R: {np.min(r)}, Max R: {np.max(r)}",
            f"Min G: {np.min(g)}, Max G: {np.max(g)}",
            f"Min B: {np.min(b)}, Max B: {np.max(b)}",
        ]

        # Grayscale & Intensity
        properties += [
            f"Mean Intensity: {np.mean(gray):.2f}",
            f"Contrast (std gray): {np.std(gray):.2f}",
            f"Brightness (mean gray): {np.mean(gray):.2f}",
            f"Entropy: {img.entropy():.2f}",
        ]

        # Compression estimation
        width, height = img.size
        uncompressed_size = width * height * 3 / 1024
        properties += [
            f"Uncompressed Size: {uncompressed_size:.2f} KB",
            f"Estimated PNG Size: {uncompressed_size * 0.6:.2f} KB",
            f"Estimated JPEG Size: {uncompressed_size * 0.3:.2f} KB",
        ]

        # Redundancy estimate
        redundancy = 100 - (np.std(gray) / 128 * 100)
        properties.append(f"Redundancy Estimate: {redundancy:.2f}%")

        # Sharpness: using Laplacian variance
        import cv2
        image_cv = cv2.imread(img_path)
        gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_cv, cv2.CV_64F).var()
        properties.append(f"Sharpness (Laplacian Var): {lap_var:.2f}")

        return properties

    except Exception as e:
        return [f"Error extracting properties: {e}"]


class ImageAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analysis App")
        self.setGeometry(100, 100, 900, 700)
        self.image_path = None
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
        """)

        self.label = QLabel("1. Upload an Image")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.btn_upload = QPushButton("Choose Image")
        self.btn_upload.clicked.connect(self.choose_image)
        self.layout.addWidget(self.btn_upload, alignment=Qt.AlignCenter)

        self.image_display = QLabel()
        self.image_display.setFixedSize(400, 400)
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("border: 2px dashed #888;")
        self.layout.addWidget(self.image_display, alignment=Qt.AlignCenter)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

    def choose_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

            # Analyze image
            props = extract_image_properties(self.image_path)
            self.result_text.setText('\n'.join(props))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageAnalysisApp()
    win.show()
    sys.exit(app.exec_())
