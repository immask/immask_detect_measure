import collections
import copy
import cv2
import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

# This is for categorizing each point with its respective facial feature.
preds_type = collections.namedtuple('predsiction_type', ['slice', 'color'])
preds_types = {'face': preds_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': preds_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': preds_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': preds_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': preds_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': preds_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': preds_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': preds_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': preds_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))}

class Detector:
	def __init__(self, image):
		self.image = image
		self.features_2d = {}
		self.features_3d = {}

	@classmethod
	def load_image_file(cls, file):
		image = cv2.imread(file)
		if image.shape[0] == image.shape[1] and image.shape[0] > 720:
			image = cv2.resize(image, (720, 720))
		else:
			image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
		return cls(image)
	
	@classmethod
	def load_image(cls, image):
		if image.shape[0] == image.shape[1] and image.shape[0] > 720:
			image = cv2.resize(image, (720, 720))
		else:
			image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
		return cls(image)

	def detect_2d(self):
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
										  flip_input=False,
										  device='cpu')
		preds = fa.get_landmarks(self.image)
		if preds is None or len(preds) != 1:
			return
		key_list = list(preds_types.keys())
		value_list = list(preds_types.values())
		for count, p_t in enumerate(preds_types.values()):
			label_list = list(range(p_t.slice.start + 1, p_t.slice.stop + 1))
			self.features_2d[key_list[value_list.index(p_t)]] = {'pts': preds[0][p_t.slice], 'labels': label_list}

	def detect_3d(self):
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, 
										  flip_input=False,
										  device='cpu')
		preds = fa.get_landmarks(self.image)
		if preds is None or len(preds) != 1:
			return
		key_list = list(preds_types.keys())
		value_list = list(preds_types.values())
		for count, p_t in enumerate(preds_types.values()):
			label_list = list(range(p_t.slice.start + 1, p_t.slice.stop + 1))
			self.features_3d[key_list[value_list.index(p_t)]] = {'pts': preds[0][p_t.slice], 'labels': label_list}

	def plot(self, sel='2d'):
		if self.features_2d and sel == '2d':
			copy_image = copy.deepcopy(self.image)

			for feat in self.features_2d:
				feat_pts = self.features_2d[feat]['pts']
				for pt in feat_pts:
					copy_image = cv2.circle(copy_image, (pt[0], pt[1]), 2, (255, 0, 0), 2)
			
			cv2.imshow('2d', copy_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		if self.features_3d and sel == '3d': 
			ax = plt.axes(projection='3d')
			
			# Data for three-dimensional scattered points
			for feat in self.features_3d:
				feat_pts = self.features_3d[feat]['pts']
				for pt in feat_pts:
					ax.scatter3D(pt[0], pt[1], pt[2], c='Black')
			plt.show()

def main():
	d = Detector.load_image_file('img/andy_selfie.jpg')
	d.detect_2d()
	print(d.features_2d)
	d.plot('2d')

if __name__ == "__main__":
	main()
