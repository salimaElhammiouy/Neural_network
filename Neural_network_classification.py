import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np 
from sklearn.neural_network import MLPClassifier
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QAction, QFileDialog, QApplication)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(496, 398)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_titre = QtWidgets.QLabel(self.centralwidget)
        self.label_titre.setGeometry(QtCore.QRect(140, 30, 211, 41))
        self.label_titre.setStyleSheet("font: 18pt \"Tw Cen MT Condensed Extra Bold\";\n"
"color: rgb(0, 0, 127);")
        self.label_titre.setObjectName("label_titre")
        self.Button_addImage = QtWidgets.QPushButton(self.centralwidget)
        self.Button_addImage.setGeometry(QtCore.QRect(190, 90, 111, 31))
        self.Button_addImage.setStyleSheet("font: 9pt \"Tw Cen MT Condensed Extra Bold\";\n"
"color:rgb(0, 0, 255);")
        self.Button_addImage.setObjectName("Button_addImage")
        self.labe_Image = QtWidgets.QLabel(self.centralwidget)
        self.labe_Image.setGeometry(QtCore.QRect(170, 160, 161, 131))
        self.labe_Image.setText("")
        self.labe_Image.setObjectName("labe_Image")
        self.label_result = QtWidgets.QLabel(self.centralwidget)
        self.label_result.setGeometry(QtCore.QRect(120, 340, 291, 31))
        self.label_result.setStyleSheet("font: 10pt \"Tw Cen MT Condensed Extra Bold\";\n"
"color: black;")
        self.label_result.setText("")
        self.label_result.setObjectName("label_result")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.Button_addImage.clicked.connect(self.openFile)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Reconnaissance des images"))
        self.label_titre.setText(_translate("MainWindow", "Le Classifieur Neuronal PMC"))
        self.Button_addImage.setText(_translate("MainWindow", "Choisir Image"))
    
  
      
    def openFile(self):
        nom_fichier = QFileDialog.getOpenFileName(None, 'Open file', '', "Image files (*.BMP *.jpg *.gif *.png)")
        self.path = nom_fichier[0]
        pathx = self.path
        pixmap = QtGui.QPixmap(pathx)

        self.labe_Image.setPixmap(pixmap)
        self.labe_Image.setScaledContents(1)
        img = cv2.imread(self.path,0)
          
        th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        Xi=th2.flatten()
        self.label_result.setText("La lettre  : =====> "+str(cl.predict([Xi])))
 

def Apprentissage():
       d=[]
       for i in range(1,63):
            img = "C:/Users/hp/Desktop/master-m2/TechniqueDD/ReseauDeNeurons1/images/images/apprentissage/".__add__(i.__str__()).__add__(".png")
            
            img = plt.imread(img,0)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                       
            ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            img=np.array(img)               
            Xi=img.flatten()   
            d.append(Xi)
       print(d)
       alphabet = ["A","A","A","A","A", "B","B","B", "C","C", "D","D", "E","E","E", "F","F", "G","G", "H","H", "I","I","I","I","I", "J","J", "K", "K", "L", "L", "M", "M", "N","N", "O","O","O", "P","P", "Q","Q", "R","R", "S","S", "T","T",
                 "U","U","U", "V","V", "W","W", "X","X", "Y","Y", "Z","Z"]
#l'apprentissage:
#nombre des couche caché dans l apprentisage par defaut 1
#nombre d norons par defaut 100
       cl=MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', alpha=0.02,
                    solver='sgd', random_state=1,max_iter=1000,
                    learning_rate_init=0.5)
       cl.fit(d,alphabet)
       E=cl.loss_curve_
 #la courbe d'erreurs:
       tauxApp=[]
       predected=cl.predict(d)
       for i in range(0,len(alphabet)):
         if alphabet[i] == predected[i]:
            tauxApp.append(predected[i])
       print("Taux d'Apprentissage des images :",(len(tauxApp)/len(alphabet))*100)
       plt.plot(E)
       plt.title("la courbe EQM en fonction du NI")
       plt.xlabel("nombre d'iterations")
       plt.ylabel("Erreur Quadratique Moyenne")
       plt.show()  
       return cl

def taux_Test():
      d=[]
      for i in range(1,42):
            img = "C:/Users/hp/Desktop/master-m2/TechniqueDD/ReseauDeNeurons1/images/images/TEST/".__add__(i.__str__()).__add__(".png")
            img = plt.imread(img,0)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                       
            ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            img=np.array(img)               
            Xi=img.flatten()
            d.append(Xi)
      TEST = ["A","A",
        "B","B", 
        "C", "C", 
        "D", "D",
        "E" ,"E",
        "F", "F",
        "G", "H", 
        "H" ,"I" ,
        "I" ,"J",
        "J", "K", 
        "L" ,"M",
        "M" ,"N", 
        "N" ,"O",
        "O" ,"P",
        "P" ,"Q",
        "R" ,"S",
        "S" ,"T",
        "U" ,"U",
        "V" ,"W",
        "X", "Y", "Z",]
      predected=cl.predict(d)
      tauxT=[]
      for i in range(0,len(TEST)):
          if TEST[i] == predected[i]:
              tauxT.append(predected[i])
      print("Taux d'apprentissage pour les donnees du Test",(len(tauxT)/len(TEST))*100,"%")

if __name__ == "__main__":
    import sys
    cl=Apprentissage()
    taux_Test()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_()) 


