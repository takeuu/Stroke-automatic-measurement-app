import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog, QHBoxLayout, QStatusBar, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PIL import Image
from ultralytics import YOLO

class VideoThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap, fps=10):
        super().__init__()
        self.cap = cap
        self.running = True
        self.fps = fps

    def run(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                self.update_frame_signal.emit(frame)
            elapsed_time = time.time() - start_time
            time.sleep(max(1.0 / self.fps - elapsed_time, 0))

    def stop(self):
        self.running = False
        self.wait()

class RulerMeasurementApp(QWidget):
    def __init__(self):
        super().__init__()

        self.mm_per_pixel = None
        self.zoom_level = 1.0
        self.inference_interval = 1
        self.last_inference_time = 0
        self.known_object_width_mm = 85.6
        self.length_history = []  # 長さの履歴を保存するリスト
        
        self.camera_id = 0
        self.cap = None
        self.thread = None

        self.initUI()
        self.open_camera(self.camera_id)

        # YOLOv8モデルの読み込み
        self.model = YOLO('yolov8n.pt')

    def open_camera(self, camera_id):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.status_bar.showMessage("カメラを開くことができませんでした。")
            return

        self.thread = VideoThread(self.cap, fps=10)
        self.thread.update_frame_signal.connect(self.process_frame)
        self.thread.start()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        control_layout = QHBoxLayout()

        self.calibrate_button = QPushButton('手動キャリブレーション', self)
        self.calibrate_button.clicked.connect(self.calibrate)
        control_layout.addWidget(self.calibrate_button)

        self.auto_calibrate_button = QPushButton('自動キャリブレーション', self)
        self.auto_calibrate_button.clicked.connect(self.auto_calibrate)
        control_layout.addWidget(self.auto_calibrate_button)

        self.measure_button = QPushButton('測定', self)
        self.measure_button.clicked.connect(self.measure_length)
        control_layout.addWidget(self.measure_button)

        self.camera_switch_button = QPushButton('カメラ切替', self)
        self.camera_switch_button.clicked.connect(self.switch_camera)
        control_layout.addWidget(self.camera_switch_button)

        self.layout.addLayout(control_layout)

        self.result_label = QLabel('長さ: ', self)
        self.layout.addWidget(self.result_label)

        # 長さ履歴表示用のスクロールエリア
        self.history_area = QScrollArea(self)
        self.history_label = QLabel(self)
        self.history_area.setWidget(self.history_label)
        self.history_area.setWidgetResizable(True)
        self.layout.addWidget(self.history_area)

        self.setLayout(self.layout)
        self.setWindowTitle('物体認識と長さ測定アプリ')

        self.status_bar = QStatusBar(self)
        self.layout.addWidget(self.status_bar)

        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(400, 300)
        self.show()

    def switch_camera(self):
        self.camera_id = 1 - self.camera_id
        self.open_camera(self.camera_id)

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_inference_time >= self.inference_interval:
            frame = self.detect_and_measure_objects(frame)
            self.last_inference_time = current_time

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_image(frame_rgb)

    def detect_and_measure_objects(self, frame):
        if self.mm_per_pixel is None:
            self.status_bar.showMessage("キャリブレーションが完了していません。")
            return frame

        # PIL形式に変換
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # YOLOv8で物体検出
        results = self.model(pil_img)

        # 出力結果の解析
        detections = results[0].boxes.xyxy.numpy()
        confs = results[0].boxes.conf.numpy()
        classes = results[0].boxes.cls.numpy()

        for (x1, y1, x2, y2), conf, cls in zip(detections, confs, classes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = self.model.names[int(cls)]

            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 長さの測定
                if self.mm_per_pixel is not None:
                    length_mm = (x2 - x1) * self.mm_per_pixel
                    cv2.putText(frame, f'{length_mm:.2f} mm', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.length_history.append(f'{label}: {length_mm:.2f} mm')
                    self.update_history()

        return frame

    def update_history(self):
        self.history_label.setText("\n".join(self.length_history))

    def display_image(self, img):
        qformat = QImage.Format_RGB888
        img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.image_label.setPixmap(QPixmap.fromImage(img))
        self.image_label.setScaledContents(True)

    def calibrate(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_bar.showMessage("キャリブレーションのための画像キャプチャに失敗しました。")
            return

        known_length, ok = QInputDialog.getDouble(self, 'キャリブレーション', '既知の長さをmm単位で入力してください:')
        if ok and known_length > 0:
            edges = cv2.Canny(frame, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

            if lines is not None:
                max_line = max(lines, key=lambda line: np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2))
                x1, y1, x2, y2 = max_line[0]
                pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                self.mm_per_pixel = known_length / pixel_distance
                self.status_bar.showMessage(f"キャリブレーション完了: {self.mm_per_pixel:.4f} mm/ピクセル")
            else:
                self.status_bar.showMessage("キャリブレーション用の線が検出されませんでした。")
        else:
            self.status_bar.showMessage("キャリブレーションがキャンセルまたは失敗しました。")

    def auto_calibrate(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_bar.showMessage("自動キャリブレーションのための画像キャプチャに失敗しました。")
            return

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.model(pil_img)

        detections = results[0].boxes.xyxy.numpy()
        classes = results[0].boxes.cls.numpy()

        for (x1, y1, x2, y2), cls in zip(detections, classes):
            if self.model.names[int(cls)] == "cell phone":
                self.mm_per_pixel = self.known_object_width_mm / (x2 - x1)
                self.status_bar.showMessage(f"自動キャリブレーション完了: {self.mm_per_pixel:.4f} mm/ピクセル")
                break

    def measure_length(self):
        ret, frame = self.cap.read()
        if not ret or self.mm_per_pixel is None:
            self.status_bar.showMessage("写真のキャプチャまたはキャリブレーションが未完了です。")
            return

        edges = cv2.Canny(frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        if lines is not None:
            max_line = max(lines, key=lambda line: np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2))
            x1, y1, x2, y2 = max_line[0]
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            length_mm = pixel_distance * self.mm_per_pixel
            self.result_label.setText(f'長さ: {length_mm:.2f} mm')
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.display_image(frame)
            self.status_bar.showMessage(f"測定された長さ: {length_mm:.2f} mm")
            self.length_history.append(f'線分: {length_mm:.2f} mm')
            self.update_history()
        else:
            self.result_label.setText("測定可能な物体が見つかりませんでした。")
            self.status_bar.showMessage("測定可能な物体が見つかりませんでした。")

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        if self.cap:
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RulerMeasurementApp()
    sys.exit(app.exec_())
