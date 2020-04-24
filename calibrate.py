import cv2
import numpy as np
import pickle

class Calibrate:
	def __init__(self, name='', image=None, device=''):   
		self.name = name
		self.image = image
		self.device = device
		self.mtx = None
		self.dist = None
		self.obj_pts = []
		self.img_pts = []

	# From the OpenCV documentation
	def get_obj_pts(self, show_img=False):
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		objp = np.zeros((8*7, 3), np.float32)
		objp[:, :2] = np.mgrid[0:7, 0:8].T.reshape(-1,2)

		ret, corners = cv2.findChessboardCorners(self.image, (7, 8), None)

		if ret:
			corners2 = cv2.cornerSubPix(self.image, corners, (11, 11), (-1, -1), criteria)
			self.obj_pts.append(objp)
			self.img_pts.append(corners2)
			cv2.drawChessboardCorners(self.image, (7,8), corners2, ret)
		
		if show_img:
			cv2.imshow('Chessboard', self.image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	def calibrate(self):
		ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(self.obj_pts, self.img_pts, self.image.shape[::-1], None, None)

	def save_info(self):
		calibrate_param = {'mtx': self.mtx, 'dist': self.dist}
		filename = './camera_correction/' + self.device + '_' + 'calibrate_param.pickle'
		pickle.dump(calibrate_param, open(filename, "wb"))
	
	def undistort(self, img_to_undistort, show_img=True):
		img = cv2.imread(img_to_undistort)
		img = cv2.resize(img, (720, 720))

		h, w = img.shape[:2]
		
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
		dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
		img_to_show = np.hstack((img, dst))

		if show_img:
			cv2.imshow('Undistorted', img_to_show)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	@classmethod
	def load_chessboard_image(cls, img_file='chessboard_calibrate.jpg', device=''): 
		img = cv2.imread(img_file)
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_img = cv2.resize(gray_img, (720, 720))
		return cls(name='chessboard', image=gray_img, device=device)
	
	@classmethod
	def load_qr_image(cls, img_file='chessboard_calibrate.jpg', device=''):
		img = cv2.imread(img_file)
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_img = cv2.resize(gray_img, (720, 720))
		return cls(name='qr', image=gray_img, device=device)

def main():
	cb = Calibrate.load_chessboard_image(img_file='chessboard_calibrate.jpg', device='iPhone_8')
	cb.get_obj_pts()
	cb.calibrate()
	cb.save_info()

if __name__ == "__main__":
	main()