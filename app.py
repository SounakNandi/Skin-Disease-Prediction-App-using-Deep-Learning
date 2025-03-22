import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import tensorflow as tf
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.setwindow()
        self.setLabel()
        self.setButton()

    def setwindow(self):
        self.setWindowTitle('Something-something.AI')

        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.windowWidth, self.windowHeight = screen_geometry.width(), screen_geometry.height()
        self.windowRatio = self.windowWidth//self.windowHeight

        self.resize(self.windowWidth // 1.5, self.windowHeight // 1.5)
        self.move(
            (self.windowWidth - self.width()) // 2,
            (self.windowHeight - self.height()) // 2
        )

        self.setMinimumSize(self.windowWidth // 1.5, self.windowHeight // 1.5)
       
    def setLabel(self):
        self.label1 = QLabel("",self)
        self.label2 = QLabel('',self)

        self.label1.setStyleSheet('''background-color: #ffffff;
                             font-weight: bold;
                             padding: 0px;
                             margin: 0px''')
        # self.label2.setStyleSheet('background-color: #666666')

        self.label1.setMaximumHeight(200)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label1.setFont(QFont('Arial', 40))


        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)

        self.central_widget.setLayout(layout)

    def setButton(self):
        self.button = QPushButton('UPLOAD',self)
        self.buttonWidth, self.buttonHeight = self.width()//2, self.height()//2
        self.button.setGeometry(100,
                           300,
                           self.buttonWidth, 
                           self.buttonHeight)
        self.button.setStyleSheet('''background-color: none;
                             border: 5px dashed grey;
                             border-radius: 20px;
                             font-weight: bold;
                             font-size: 100px''')
        self.button.clicked.connect(self.upload_image)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Upload Image", "", "Images (*.png *.xpm *.jpg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.label2.setPixmap(pixmap.scaled(250, 250))
            model = tf.keras.models.load_model('model.h5')
            self.prediction(model,file_path)
    
    def prediction(self, model, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(200,200))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        class_names = ['benign', 'malignant']

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        self.label1.setText(f'predicted class: {predicted_class}\nconfidence: {confidence}')
        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
