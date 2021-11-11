import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np 
from sklearn.neural_network import MLPClassifier
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import timeit

from PyQt5 import QtCore, QtGui, QtWidgets

y=[]
# l image a laquel en vas faire du teste
X=[]

gauss=''
ber=''
multi=''

for i in range(1,63):
    img = cv2.imread("C:/Users/hp/Desktop/master-m2/TechniqueDD/ReseauDeNeurons1/images/images/apprentissage/".__add__(i.__str__()).__add__(".png"),0)
    th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    Xi=th2.flatten()
    y.append(Xi)
    

#l apprentissage avec gaussien 
    def gaussien(d):
        global gauss
        gauss = GaussianNB(var_smoothing=0.12)#Portion de la plus grande variance de toutes les 
        #entités qui est ajoutée aux variances pour la stabilité des calculs.
        gauss.fit(y,d)
        return gauss

#l apprentissage evec bernoulli  
    def bernoulli(d):
        global ber
        ber = BernoulliNB(alpha=0.5, binarize=0.42)#alpha parm de lissage 1.0 par defaut
        ber.fit(y,d)
        return ber


#l apprentissage avec multinomial  
    def multinomial(d):
        global multi
        multi = MultinomialNB(alpha=0.9)
        multi.fit(y,d)
        return multi

d=["A","A","A","A","A",
                    "B","B","B",
                    "C","C",
                    "D","D",
                    "E","E","E",
                    "F","F",
                    "G","G",
                    "H","H",
                    "I","I","I","I","I",
                    "J","J",
                    "K","K",
                    "L","L",
                    "M","M",
                    "N","N",
                    "O","O","O",
                    "P", "P",
                    "Q","Q",
                    "R","R",
                    "S", "S",
                    "T","T",
                    "U","U","U",
                    "V", "V",
                    "W","W",
                    "X","X",
                    "Y","Y",
                    "Z","Z"]
#appel des fonctions
gaussien(d)
bernoulli(d)
multinomial(d)

def taux_recA(methode):
   alphabet = ["A","A","A","A","A", "B","B","B", "C","C", "D","D", "E","E","E", "F","F", "G","G", "H","H", "I","I","I","I","I", "J","J", "K", "K", "L", "L", "M", "M", "N","N", "O","O","O", "P","P", "Q","Q", "R","R", "S","S", "T","T",
                 "U","U","U", "V","V", "W","W", "X","X", "Y","Y", "Z","Z"]
   s=0
   for i in range(1,63):
        img = cv2.imread("C:/Users/hp/Desktop/master-m2/TechniqueDD/ReseauDeNeurons1/images/images/apprentissage/".__add__(i.__str__()).__add__(".png"),0)
        th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        Ki=th2.flatten()
        if(alphabet[i-1]==methode.predict([Ki])):
            s=s+1

   taux_A = s/62*100
   return taux_A
#la reconnaissance
def taux_recT(methode):
    te=["A" ,"A",
                    "B" ,"B", 
                    "C" ,"C", 
                    "D" ,"D",
                    "E" ,"E",
                    "F" ,"F",
                    "G" ,"H", 
                    "H" ,"I",
                    "I" ,"J",
                    "J" ,"K",
                    "L" ,"M",
                    "M" ,"N", 
                    "N" ,"O",
                    "O" ,"P",
                    "P" ,"Q",
                    "R" ,"S",
                    "S" ,"T",
                    "U" ,"U",
                    "V" ,"W",
                    "X" ,"Y",
                    "Z" ,   ]
    s=0
    for i in range(1,42):
        img = cv2.imread("C:/Users/hp/Desktop/master-m2/TechniqueDD/ReseauDeNeurons1/images/images/TEST/".__add__(i.__str__()).__add__(".png"),0)
        th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        Ki=th2.flatten()
        if(te[i-1]==methode.predict([Ki])):
            s=s+1

    taux_app = s/41*100
    return taux_app
    
    
def temp_Apprendre():     
    temp_gauss = timeit.Timer('gaussien(d)','from __main__ import gaussien, d')
    temp_ber = timeit.Timer('bernoulli(d)','from __main__ import bernoulli, d')
    temp_multi = timeit.Timer('multinomial(d)','from __main__ import multinomial, d')
    a=temp_gauss.timeit(1)
    b=temp_ber.timeit(1)
    c=temp_multi.timeit(1)
    return a,b,c
 
  

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(998, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(200, 30, 581, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(80, 110, 831, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(80, 150, 821, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(80, 190, 821, 21))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(80, 240, 821, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(80, 290, 831, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(73, 120, 20, 181))
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setGeometry(QtCore.QRect(183, 120, 20, 181))
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 130, 91, 21))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(90, 170, 91, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(90, 210, 91, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(90, 255, 91, 31))
        self.label_5.setObjectName("label_5")
        self.line_8 = QtWidgets.QFrame(self.centralwidget)
        self.line_8.setGeometry(QtCore.QRect(333, 120, 21, 181))
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(200, 130, 131, 21))
        self.label_6.setObjectName("label_6")
        self.label_t1 = QtWidgets.QLabel(self.centralwidget)
        self.label_t1.setGeometry(QtCore.QRect(200, 160, 131, 31))
        self.label_t1.setObjectName("label_t1")
        self.label_t2 = QtWidgets.QLabel(self.centralwidget)
        self.label_t2.setGeometry(QtCore.QRect(200, 210, 131, 31))
        self.label_t2.setObjectName("label_t2")
        self.label_t3 = QtWidgets.QLabel(self.centralwidget)
        self.label_t3.setGeometry(QtCore.QRect(200, 260, 131, 31))
        self.label_t3.setObjectName("label_t3")
        self.line_9 = QtWidgets.QFrame(self.centralwidget)
        self.line_9.setGeometry(QtCore.QRect(580, 120, 20, 181))
        self.line_9.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.line_10 = QtWidgets.QFrame(self.centralwidget)
        self.line_10.setGeometry(QtCore.QRect(790, 120, 20, 181))
        self.line_10.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.line_11 = QtWidgets.QFrame(self.centralwidget)
        self.line_11.setGeometry(QtCore.QRect(893, 120, 20, 181))
        self.line_11.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(350, 120, 231, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(600, 120, 181, 31))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(810, 130, 81, 21))
        self.label_9.setObjectName("label_9")
        self.label_ap1 = QtWidgets.QLabel(self.centralwidget)
        self.label_ap1.setGeometry(QtCore.QRect(350, 160, 231, 31))
        self.label_ap1.setObjectName("label_ap1")
        self.label_ap2 = QtWidgets.QLabel(self.centralwidget)
        self.label_ap2.setGeometry(QtCore.QRect(350, 200, 231, 41))
        self.label_ap2.setObjectName("label_ap2")
        self.label_ap3 = QtWidgets.QLabel(self.centralwidget)
        self.label_ap3.setGeometry(QtCore.QRect(350, 260, 231, 31))
        self.label_ap3.setObjectName("label_ap3")
        self.label_te1 = QtWidgets.QLabel(self.centralwidget)
        self.label_te1.setGeometry(QtCore.QRect(600, 160, 191, 31))
        self.label_te1.setObjectName("label_te1")
        self.label_te2 = QtWidgets.QLabel(self.centralwidget)
        self.label_te2.setGeometry(QtCore.QRect(600, 200, 191, 31))
        self.label_te2.setObjectName("label_te2")
        self.label_te3 = QtWidgets.QLabel(self.centralwidget)
        self.label_te3.setGeometry(QtCore.QRect(590, 250, 201, 31))
        self.label_te3.setObjectName("label_te3")
        self.result1 = QtWidgets.QLabel(self.centralwidget)
        self.result1.setGeometry(QtCore.QRect(810, 170, 71, 21))
        self.result1.setObjectName("result1")
        self.result2 = QtWidgets.QLabel(self.centralwidget)
        self.result2.setGeometry(QtCore.QRect(810, 210, 81, 31))
        self.result2.setObjectName("result2")
        self.result3 = QtWidgets.QLabel(self.centralwidget)
        self.result3.setGeometry(QtCore.QRect(810, 260, 71, 31))
        self.result3.setObjectName("result3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(420, 320, 211, 41))
        self.pushButton.setObjectName("pushButton")
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(390, 390, 271, 191))
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Le réseau bayésien naïf"))
        self.label_2.setText(_translate("MainWindow", "Classifier"))
        self.label_3.setText(_translate("MainWindow", "Gaussian NB"))
        self.label_4.setText(_translate("MainWindow", "multinomial NB"))
        self.label_5.setText(_translate("MainWindow", "Bernoulli NB"))
        self.label_6.setText(_translate("MainWindow", "Temps d\'execution"))
        self.label_t1.setText(_translate("MainWindow", "TextLabel"))
        self.label_t2.setText(_translate("MainWindow", "TextLabel"))
        self.label_t3.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "Taux de reconnaissance sur l\'apprentissage"))
        self.label_8.setText(_translate("MainWindow", "taux de reconnaissance sur le test"))
        self.label_9.setText(_translate("MainWindow", "resultat"))
        self.label_ap1.setText(_translate("MainWindow", "TextLabel"))
        self.label_ap2.setText(_translate("MainWindow", "TextLabel"))
        self.label_ap3.setText(_translate("MainWindow", "TextLabel"))
        self.label_te1.setText(_translate("MainWindow", "TextLabel"))
        self.label_te2.setText(_translate("MainWindow", "TextLabel"))
        self.label_te3.setText(_translate("MainWindow", "TextLabel"))
        self.result1.setText(_translate("MainWindow", ""))
        self.result2.setText(_translate("MainWindow", ""))
        self.result3.setText(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "choisir image"))
        self.label_ap1.setText(str(taux_recA(gauss)))
        self.label_ap2.setText(str(taux_recA(multi)))
        self.label_ap3.setText(str(taux_recA(ber)))
        self.label_te1.setText(str(taux_recT(gauss)))
        self.label_te2.setText(str(taux_recT(multi)))
        self.label_te3.setText(str(taux_recT(ber)))
        self.pushButton.clicked.connect(self.openFile)
        self.label_t1.setText(_translate("MainWindow",str(t1)))
        self.label_t2.setText(_translate("MainWindow",str(t2)))
        self.label_t3.setText(_translate("MainWindow",str(t3)))
        
    def openFile(self):
        nom_fichier = QFileDialog.getOpenFileName(None, 'Open file', '', "Image files (*.BMP *.jpg *.gif *.png)")
        self.path = nom_fichier[0]
        pathx = self.path
        pixmap = QtGui.QPixmap(pathx)
    
        self.label_image.setPixmap(pixmap)
        self.label_image.setScaledContents(1)
        img = cv2.imread(self.path,0)
              
        th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        Xi=th2.flatten()
        self.result1.setText(str(gauss.predict([Xi])))
        self.result2.setText(str(multi.predict([Xi])))
        self.result3.setText(str(ber.predict([Xi])))
    
if __name__ == "__main__":
    import sys
    
    t1,t2,t3= temp_Apprendre() 
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_()) 