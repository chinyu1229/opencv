import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QMainWindow
from PyQt5.QtGui import QPixmap,QImage

from PY_UI import Ui_Form

class MyMainForm(QMainWindow, Ui_Form):
	def __init__(self,parent = None):
		super(MyMainForm, self).__init__(parent)
		self.setupUi(self)
		self.pushButton1_1.clicked.connect(self.Q1_1)
		self.pushButton1_2.clicked.connect(self.Q1_2)
		self.pushButton1_3.clicked.connect(self.Q1_3)
		self.pushButton1_4.clicked.connect(self.Q1_4)
		self.pushButton2_1.clicked.connect(self.Q2_1)
		self.pushButton2_2.clicked.connect(self.Q2_2)
		self.pushButton2_3.clicked.connect(self.Q2_3)
		self.pushButton3_1.clicked.connect(self.Q3_1)
		self.pushButton3_2.clicked.connect(self.Q3_2)
		self.pushButton3_3.clicked.connect(self.Q3_3)
		self.pushButton3_4.clicked.connect(self.Q3_4)
		self.pushButton4.clicked.connect(self.Q4)


	def closeEvent(self, event):
		sys.exit(app.quit())

	def Q1_1(self):
		img = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
		cv2.namedWindow('1',0)
		height = img.shape[0]
		width = img.shape[1]
		print('Height = ',height)
		print('Width = ',width)
		cv2.imshow('1',img)
		cv2.waitKey(0)
		#cv2.destroyAllWindows()
		cv2.destroyWindow('1')

	def Q1_2(self):
		#cv2.destroyAllWindows()
		img = cv2.imread('./Dataset_opencvdl/Q1_Image/Flower.jpg')
		(b,g,r) = cv2.split(img)
		zeros = np.zeros(img.shape[:2],dtype = "uint8")
		cv2.imshow('Red',cv2.merge([zeros,zeros,r]))
		cv2.imshow('Green',cv2.merge([zeros,g,zeros]))
		cv2.imshow('Blue',cv2.merge([b,zeros,zeros]))
		cv2.waitKey(0)
		#cv2.destroyAllWindows()
		cv2.destroyWindow('Red')
		cv2.destroyWindow('Green')
		cv2.destroyWindow('Blue')
	def Q1_3(self):
		img = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
		cv2.namedWindow('Oringinal Image',0)
		cv2.imshow('Oringinal Image',img)
		flip = cv2.flip(img,1)
		cv2.namedWindow('Result',0)
		cv2.imshow('Result',flip)
		cv2.waitKey(0)
		cv2.destroyWindow('Oringinal Image')
		cv2.destroyWindow('Result')

	def Q1_4(self):
		title = 'BLENDING'
		trackbar_name = 'BLEND'
		foreground = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
		background = cv2.flip(foreground,1)		
		def on_trackbar(val):
			alpha = val / 255
			beta = ( 1.0 - alpha )
			dst = cv2.addWeighted(foreground,alpha,background,beta,0.0)
			cv2.imshow('BLENDING', dst)
		cv2.namedWindow(title)
		cv2.createTrackbar(trackbar_name,title,0,255,on_trackbar)
		on_trackbar(0)
		cv2.waitKey(0)
		cv2.destroyWindow(title)
	
	def Q2_1(self):	
		img = cv2.imread('./Dataset_opencvdl/Q2_Image/Cat.png')
		median = cv2.medianBlur(img,7)
		cv2.imshow('median',median)
		cv2.waitKey(0)
		cv2.destroyWindow('median')
	
	def Q2_2(self):	
		img = cv2.imread('./Dataset_opencvdl/Q2_Image/Cat.png')
		gaussian = cv2.GaussianBlur(img,(3,3),0)
		cv2.imshow('gaussian',gaussian)
		cv2.waitKey(0)
		cv2.destroyWindow('gaussian')	

	def Q2_3(self):	
		img = cv2.imread('./Dataset_opencvdl/Q2_Image/Cat.png')
		bilateral = cv2.bilateralFilter(img,9,90,90)
		cv2.imshow('bilateral',bilateral)
		cv2.waitKey(0)
		cv2.destroyWindow('bilateral')
		
	def Q3_1(self):
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg',0)
		a,b = np.mgrid[-1:2,-1:2]
		gaussian_kernel = np.exp(-(a**2 + b**2))
		#normalize
		gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
		gaussian_kernel = np.around(gaussian_kernel,3)
		y,x = img.shape
		y = y - 3 + 1
		x = x - 3 + 1
		result = np.zeros((y,x))
		for i in range(y):
			for j in range(x):
				result[i][j] = np.sum(img[i:i+3,j:j+3]*gaussian_kernel)
		plt.figure('guassian blur')		
		plt.imshow(result,cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
	def Q3_2(self):
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg',0)
		a,b = np.mgrid[-1:2,-1:2]
		gaussian_kernel = np.exp(-(a**2 + b**2))
		#normalize
		gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
		gaussian_kernel = np.around(gaussian_kernel,3)
		y,x = img.shape
		y = y - 3 + 1
		x = x - 3 + 1
		result = np.zeros((y,x))
		for i in range(y):
			for j in range(x):
				result[i][j] = np.sum(img[i:i+3,j:j+3]*gaussian_kernel)
		
		r,s = result.shape
		r = r - 3 + 1
		s = s - 3 + 1
		sobel_x = np.zeros((r,s))
		for i in range(2,r,1):
			for j in range(2,s,1):
				x_result = abs(result[i-1][j+1] + 2 * result[i][j+1] + result[i+1][j+1] - result[i-1][j-1] - 2 * result[i][j-1] - result[i+1][j-1])
				if x_result > 255:
					sobel_x[i][j] = x_result % 255 
				else:
					sobel_x[i][j] = x_result
		plt.figure('sobel_x')
		plt.imshow(sobel_x,cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
	def Q3_3(self):
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg',0)
		a,b = np.mgrid[-1:2,-1:2]
		gaussian_kernel = np.exp(-(a**2 + b**2))
		#normalize
		gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
		gaussian_kernel = np.around(gaussian_kernel,3)
		y,x = img.shape
		y = y - 3 + 1
		x = x - 3 + 1
		result = np.zeros((y,x))
		for i in range(y):
			for j in range(x):
				result[i][j] = np.sum(img[i:i+3,j:j+3]*gaussian_kernel)
		
		r,s = result.shape
		r = r - 3 + 1
		s = s - 3 + 1
		sobel_y = np.zeros((r,s))
		for i in range(2,r,1):
			for j in range(2,s,1):
				y_result = abs(result[i-1][j-1] + 2 * result[i-1][j] + result[i-1][j+1] - result[i+1][j-1] - 2 * result[i+1][j] - result[i+1][j+1])
				if y_result > 255:
					sobel_y[i][j] = y_result % 255 
				else:
					sobel_y[i][j] = y_result
		plt.figure('sobel_y')
		plt.imshow(sobel_y,cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()

	def Q3_4(self):
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg',0)
		a,b = np.mgrid[-1:2,-1:2]
		gaussian_kernel = np.exp(-(a**2 + b**2))
		#normalize
		gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
		gaussian_kernel = np.around(gaussian_kernel,3)
		y,x = img.shape
		y = y - 3 + 1
		x = x - 3 + 1
		result = np.zeros((y,x))
		for i in range(y):
			for j in range(x):
				result[i][j] = np.sum(img[i:i+3,j:j+3]*gaussian_kernel)
		
		r,s = result.shape
		r = r - 3 + 1
		s = s - 3 + 1
		magnitude = np.zeros((r,s))
		for i in range(2,r,1):
			for j in range(2,s,1):
				x_result = abs(result[i-1][j+1] + 2 * result[i][j+1] + result[i+1][j+1] - result[i-1][j-1] - 2 * result[i][j-1] - result[i+1][j-1])
				y_result = abs(result[i-1][j-1] + 2 * result[i-1][j] + result[i-1][j+1] - result[i+1][j-1] - 2 * result[i+1][j] - result[i+1][j+1])
				sobel = np.sqrt(x_result**2 + y_result**2)
				if sobel > 255:
					magnitude[i][j] = sobel % 255 
				else:
					magnitude[i][j] = sobel
		plt.figure('Magnitude')
		plt.imshow(magnitude,cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()

	def Q4(self):
		angle = self.Rotation.text()
		scale_val = self.Scaling.text()
		Xnew = self.Tx.text()
		Ynew = self.Ty.text()
		img = cv2.imread('./Dataset_opencvdl/Q4_Image/Parrot.png')
		r,c = img.shape[:2]

		trans = np.float32([[1,0,Xnew],[0,1,Ynew]])
		trans_ = cv2.warpAffine(img,trans,(c,r))

		rotation = cv2.getRotationMatrix2D((160+int(Xnew),84+int(Ynew)),float(angle),float(scale_val))
		res = cv2.warpAffine(trans_,rotation,(c,r))
		#plt.imshow(res)
		#plt.show()
		cv2.imshow('Original Image',img)
		cv2.imshow('Image RST',res)
		cv2.waitKey(0)
		cv2.destroyWindow('Original Image')
		cv2.destroyWindow('Image RST')
		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	myWin = MyMainForm()
	myWin.show()
	sys.exit(app.exec_())
	

