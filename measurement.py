import numpy as np
import math
from stl import mesh
import trimesh

# Points 69, 70, 71 are for one cheek, and points 79, 80, 81 are for the
# other cheek
label_of_pts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
                17, 28, 30, 69, 70, 71, 79, 80, 81]

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
    # features_3d to approximate points on the cheeks.
    def extract_pt_thru_calc(self, label):
        if label == 69:
            # Compute point by knowing the nostil, middle to right eye, and eyebrow.
            return
        elif label == 70:
            return
        elif label == 71:
            return
        elif label == 79:
            # Compute point by knowing the nostil, middle to left eye, and eyebrow.
            return
        elif label == 80:
            return
        elif label == 81:
            return

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
        elif label >= 69 and label < label_of_pts[-1]:
            return self.extract_pt_thru_calc(label)
        else:
            return [0, 0, 0]
