import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
import trimesh

# Points 69, 70, 71 are for one cheek, and points 79, 80, 81 are for the
# other cheek
label_of_pts = list(range(1, 82))

class Measurement:
    def __init__(self, features_3d, ref_dist=0, units='cm'):
        self.faces = []
        self.faces_construct = open('faces_construct.im')
        self.pts = []
        self.features_3d = features_3d
        self.ref_dist = ref_dist
        self.units = units      # This can also be in inches.

    def measure_mask(self):
        # Get the distance between the corner of the mouth and the tip of the nose.
        tip_nose_pt = self.get_pt(30)
        corner_mouth_pt = self.get_pt(49)
        dist_3d_space = self.distance(tip_nose_pt, corner_mouth_pt)
        ratio = self.ref_dist / dist_3d_space

    def find_all_pts(self):
        if self.pts:
            self.pts = []
        for pt in label_of_pts:
            self.pts.append(self.get_pt(pt))

    def find_all_faces(self):
        if self.faces:
            self.faces = []
        for line in self.faces_construct:
            labels = [int(label) for label in line.split(',')]
            # This is needed in order to know what the index of the labels are for constructing
            # the face. 
            self.faces.append([label_of_pts.index(labels[0]),  
                            label_of_pts.index(labels[1]), 
                            label_of_pts.index(labels[2])])

    def save_stl(self):
        pts_np = np.asarray(self.pts)
        faces_np = np.asarray(self.faces)

        # Create the mesh
        mask = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces_np):
            for j in range(3):
                mask.vectors[i][j] = pts_np[f[j],:]

        # Write the mesh to file "mask.stl"
        mask.save('stl/mask.stl')

    def preview(self):
        mesh = trimesh.load('stl/mask.stl')
        mesh.show()

    def distance(self, pt_0=[0, 0, 0], pt_1=[0, 0, 0]):
        return math.sqrt((pt_1[0] - pt_0[0])**2 + (pt_1[1] - pt_0[1])**2 + (pt_1[2] - pt_0[2])**2)  

    # The first 68 points of the face can be obtained through features_3d. 
    def extract_pt_from_feat(self, facial_feature, label):
        pts_list = self.features_3d[facial_feature]['pts']
        labels_list = self.features_3d[facial_feature]['labels']
        get_idx = labels_list.index(label)
        return pts_list[get_idx]

    # The rest of the points can be found through some point manipulation. We will still need 
    # features_3d to approximate points on the cheeks. THESE ARE NOT PART OF THE ORIGINAL 68
    # POINTS IDENTIFIED BY THE DETECTION ALGORITHM!
    def extract_pt_thru_calc(self, label):
        if label == 69:
            # Compute point by knowing the nostil, middle to right eye, and eyebrow. This is for the
            # middle of the cheek.
            return [self.get_pt(46)[0], self.get_pt(34)[1], self.get_pt(24)[2]]
        elif label == 70:
            return [self.get_pt(27)[0], self.get_pt(29)[1], self.get_pt(27)[2]]
        elif label == 71:
            return [self.get_pt(48)[0], self.get_pt(29)[1], self.get_pt(48)[2]]
        elif label == 79:
            # Compute point by knowing the nostil, middle to right eye, and eyebrow. This is for the
            # middle of the cheek.
            return [self.get_pt(37)[0], self.get_pt(34)[1], self.get_pt(21)[2]]
        elif label == 80:
            return [self.get_pt(40)[0], self.get_pt(29)[1], self.get_pt(40)[2]]
        elif label == 81:
            return [self.get_pt(18)[0], self.get_pt(29)[1], self.get_pt(18)[2]]
        else:
            return [0, 0, 0]

    def get_pt(self, label):
        if label >= 1 and label < 18:
            return self.extract_pt_from_feat('face', label)
        elif label >= 18 and label < 23:
            return self.extract_pt_from_feat('eyebrow1', label)
        elif label >= 23 and label < 28:
            return self.extract_pt_from_feat('eyebrow2', label)
        elif label >= 28 and label < 32:
            return self.extract_pt_from_feat('nose', label)
        elif label >= 32 and label < 37:
            return self.extract_pt_from_feat('nostril', label)
        elif label >= 37 and label < 43:
            return self.extract_pt_from_feat('eye1', label)
        elif label >= 43 and label < 49:
            return self.extract_pt_from_feat('eye2', label)
        elif label >= 49 and label < 61:
            return self.extract_pt_from_feat('lips', label)
        elif label >= 61 and label < 69:
            return self.extract_pt_from_feat('teeth', label)
        else:
            return self.extract_pt_thru_calc(label)

    def plot(self):
        ax = plt.axes(projection='3d')
        # Data for three-dimensional scattered points
        for feat in self.features_3d:
            feat_pts = self.features_3d[feat]['pts']
            for pt in feat_pts:
                ax.scatter3D(pt[0], pt[1], pt[2], c='Black')
        # This is point 79
        ax.scatter3D(self.get_pt(37)[0], self.get_pt(34)[1], self.get_pt(21)[2], c='Red')
        # This is point 80
        ax.scatter3D(self.get_pt(40)[0], self.get_pt(29)[1], self.get_pt(40)[2], c='Green')
        # This is point 81
        ax.scatter3D(self.get_pt(18)[0], self.get_pt(29)[1], self.get_pt(18)[2], c='Blue')
        plt.show()