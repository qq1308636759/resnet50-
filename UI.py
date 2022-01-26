from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtWidgets import QApplication,  QFileDialog
from PyQt5.QtGui import QPixmap
import main as sb
from tensorflow.keras.models import load_model
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(765, 402)
        self.centralwidget = QtWidgets.QWidget(Form)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(70, 50, 256, 256))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(560, 300, 151, 61))
        self.pushButton.setObjectName("pushButton")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(420, 50, 256, 51))
        self.textBrowser.setStyleSheet("border:0px;\n""")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 300, 151, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser_1 = QtWidgets.QTextBrowser(Form)
        self.textBrowser_1.setGeometry(QtCore.QRect(420, 140, 261, 101))
        self.textBrowser_1.setObjectName("textBrowser1")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.pushButton.clicked.connect(self.prediction)
        self.pushButton_2.clicked.connect(self.openimg)

    def openimg(self):
        self.img_file, _ = QFileDialog.getOpenFileName(self.centralwidget, 'Open file',
                                                         r'Resnet\\',
                                                         'Image files (*.jpg)')
        print(self.img_file)
        self.img = QPixmap(self.img_file)
        self.label.setPixmap(self.img)
        self.label.setScaledContents(True)
    def prediction(self):
        self.image=sb.read(self.img_file)

        model = sb.model()
        pred = sb.pre(model,self.image)
        pred = str(pred)
        self.textBrowser_1.append("<font size=\"8\" color=\"#000000\">" + '该垃圾为:' + pred + "</font>")
        QtWidgets.QApplication.processEvents()  # 防止进程卡死

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "垃圾分类"))
        self.label.setText(_translate("Form", "TextLabel"))
        self.pushButton.setText(_translate("Form", "开始识别"))
        self.textBrowser.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:20pt;\">垃圾分类系统</span></p></body></html>"))
        self.pushButton_2.setText(_translate("Form", "加载图片"))
if __name__ == '__main__':
    import PyQt5
    app = QApplication(sys.argv)
    ex = Ui_Form()
    window = PyQt5.QtWidgets.QMainWindow()
    ex.setupUi(window)
    window.show()
    sys.exit(app.exec_())