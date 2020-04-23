import collections
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

facial_feat = ['face', 'eyebrow1', 'eyebrow2', 'nose', 'nostril', 'eye1',
                      'eye2', 'lips', 'teeth']

class detect_3d:
	def __init__(self, image, name):
		self.image = image
		self.name = name
		self.features = {}

	@classmethod
	def load_image(cls, file):
		image = cv2.imread(file)
		return cls(image, file)

	def detect(self):
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
		preds = fa.get_landmarks(self.image)
		
		if preds is None or len(preds) != 1:
			return

		for count, preds_type in enumerate(preds_types.values()):
			self.features[facial_feat[count]] = preds[0][preds_type.slice]

	def plot(self):
		ax = plt.axes(projection='3d')

		# Data for three-dimensional scattered points
		for feat in self.features:
			feat_pts = self.features[feat]
			for pt in feat_pts:
				ax.scatter3D(pt[0], pt[1], pt[2], c='Black');
		plt.show()

d_3d = detect_3d.load_image('selfie.jpg')
d_3d.detect()
d_3d.plot()
