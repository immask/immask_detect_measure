import cv2

class Calibrate:
	def __init__(self, name='', image=None):   
		self.name = name
		self.image = image
		self.obj_pts = []
		self.img_pts = []

	def calibrate(self):
		return

	@classmethod
	def load_chessboard_image(cls, img_file):
		img = cv2.imread(img_file)
		return cls('chessboard', img)
	
	@classmethod
	def load_qr_image(cls, img_file):
		img = cv2.imread(img_file)
		return cls('qr', img)