import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime, timedelta
import pandas as pd
from fuzzywuzzy import fuzz
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('cnnmodel.h5')

# Define the categories (fruits and vegetables)
categories = ['apples', 'banana', 'orange', 'carrot', 'tomato',
              'strawberry', 'potato', 'bellpepper', 'cucumber', 'mango']

# Load the dataset
df = pd.read_excel('dataset.xlsx')


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.update_frame)
        self.image_captured = False

    def initUI(self):
        self.setWindowTitle('Image Capture and Prediction')
        # Set the background color
        self.setStyleSheet(
            "background-color: #DDD6F3;background-image: linear-gradient(90deg, #74EBD5 0%, #9FACE6 100%);")

        # Get the screen dimensions
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        self.layout = QVBoxLayout(self)

        self.btnCapture = QPushButton('Capture Image')
        self.btnCapture.clicked.connect(self.capture_image)
        # Set the fixed size of the capture button
        self.btnCapture.setFixedSize(700, 100)
        self.btnCapture.setStyleSheet(
            "background-color: #FAACA8; color: white; font-size: 18px; border: none")  # Set the button color

        self.frameLabel = QLabel(self)
        self.frameLabel.setAutoFillBackground(
            True)  # Enable background filling
        # Align the frame label to the center
        self.frameLabel.setAlignment(Qt.AlignCenter)

        self.imageLabel = QLabel(self.frameLabel)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        # Set the size to fit the screen
        self.imageLabel.setMinimumSize(screen_width, screen_height)

        self.layout.addStretch()
        self.layout.addWidget(self.btnCapture, alignment=Qt.AlignCenter)
        self.layout.addStretch()
        self.layout.addWidget(self.frameLabel)
        self.layout.addWidget(self.imageLabel)

        self.setLayout(self.layout)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame in the image label
            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(frame_rgb.data, width, height,
                             bytes_per_line, QImage.Format_RGB888)
            self.imageLabel.setPixmap(QPixmap.fromImage(q_image))

            # Set the image captured flag
            self.image_captured = True

            # Stop the capture timer
            self.capture_timer.stop()

            # Process the captured image
            self.process_captured_image(frame_rgb)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the image label
            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(frame_rgb.data, width, height,
                             bytes_per_line, QImage.Format_RGB888)
            self.imageLabel.setPixmap(QPixmap.fromImage(q_image))

    def process_captured_image(self, image):
        if self.image_captured:
            # Make predictions on the captured image
            predicted_category, freshness = self.predict_image(image)

            if predicted_category is not None:
                self.display_notification(predicted_category, freshness)
            else:
                self.display_notification("Unable to predict category.", "")

    def predict_image(self, image):
        image = cv2.resize(image, (64, 64))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)[0]
        predicted_label = np.argmax(prediction)
        predicted_category = categories[predicted_label // 2]
        freshness = "Fresh" if predicted_label % 2 == 0 else "Expired"
        return predicted_category, freshness

    def display_notification(self, predicted_category, freshness):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Prediction Result")
        msg_box.setText(
            f"Predicted Category: {predicted_category}\nFreshness: {freshness}")
        msg_box.exec_()

    def start_capture(self):
        # Start capturing frames every 30 milliseconds
        self.capture_timer.start(30)

    def stop_capture(self):
        self.cap.release()

    def closeEvent(self, event):
        self.cap.release()
        self.stop_capture()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    window.start_capture()
    sys.exit(app.exec_())
