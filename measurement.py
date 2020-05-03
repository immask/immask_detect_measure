from itertools import combinations
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
import svgwrite
from svgwrite import cm, mm
import trimesh

# Specify the height and width of the paper that the outline of the shapes making up
# the mask will be printed on.
PAPER_HEIGHT = "27.94cm"
PAPER_WIDTH = "21.59cm"
PAPER_SIZE = (PAPER_WIDTH, PAPER_HEIGHT)

class Measurement:
    def __init__(self, features_3d, ref_dist=0, units='cm'):
        self.faces = []
        self.faces_construct = open('faces_construct.im')   # This will help create the face using preexisting points
        self.pts = []
        self.shapes = {}
        self.features_3d = features_3d
        self.ref_dist = ref_dist
        self.units = units      # This can also be in inches.
        self.ratio = self.measure_ratio()

    def find_all_pts(self):
        if self.pts:
            self.pts = []
        label_of_pts = list(range(0, 82))
        for pt in label_of_pts:
            self.pts.append(self.get_pt(pt))

    def find_all_faces(self):
        if self.faces:
            self.faces = []
        for line in self.faces_construct:
            labels = [int(label) for label in line.split(',')]
            # This is needed in order to know what the index of the labels are for constructing
            # the face. 
            self.faces.append([labels[0], labels[1], labels[2]])
        self.faces_construct.seek(0)

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

    # The first 68 points of the face can be obtained through features_3d. 
    def extract_pt_from_feat(self, facial_feature, label):
        pts_list = self.features_3d[facial_feature]['pts']
        labels_list = self.features_3d[facial_feature]['labels']
        get_idx = labels_list.index(label)
        return pts_list[get_idx].tolist()

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
        plt.show()

    def distance(self, pt_0, pt_1):
        if len(pt_0) == 3 and len(pt_1) == 3:
            return math.sqrt((pt_1[0] - pt_0[0])**2 + (pt_1[1] - pt_0[1])**2 + (pt_1[2] - pt_0[2])**2)  
        elif len(pt_0) == 2 and len(pt_1) == 2:
            return math.sqrt((pt_1[1] - pt_0[1])**2 + (pt_1[2] - pt_0[2])**2)  

    def measure_ratio(self):
        # Get the distance between the corner of the mouth and the tip of the nose.i
        tip_nose_pt = self.get_pt(30)
        corner_mouth_pt = self.get_pt(49)
        dist_3d_space = self.distance(tip_nose_pt, corner_mouth_pt)
        
        if self.units == 'in':
            return self.ref_dist * 2.54 / dist_3d_space
        else:
            return self.ref_dist / dist_3d_space

    # We want to transform the 3D coordinates making up the triangle into 2D coordinates.
    def to_2d_trig(self, face_id):
        v_1 = self.shapes[face_id]['original'][0]
        v_2 = self.shapes[face_id]['original'][1]
        v_3 = self.shapes[face_id]['original'][2]

        h = math.sqrt((v_2[2] - v_1[2])**2 + (v_2[1] - v_1[1])**2 + (v_2[0] - v_1[0])**2)
        i = ((v_3[2] - v_1[2])*(v_2[2] - v_1[2]) + (v_3[1] - v_1[1])*(v_2[1] - v_1[1]) + (v_3[0] - v_1[0])*(v_2[0] - v_1[0]))/h
        j = math.sqrt((v_3[2] - v_1[2])**2 + (v_3[1] - v_1[1])**2 + (v_3[0] - v_1[0])**2 - i**2)

        return [0, 0], [h, 0], [i, j]

    def calculate_area(self, vertices):
        if len(vertices) != 3 or (len(vertices[0]) != len(vertices[1]) and len(vertices[0]) != len(vertices[2])):
            return -1
        
        side_1 = [val_2 - val_1 for (val_2, val_1) in zip(vertices[1], vertices[0])]
        side_2 = [val_2 - val_1 for (val_2, val_1) in zip(vertices[2], vertices[0])]
        
        dot_product = sum(val_2 * val_1 for (val_2, val_1) in zip(side_1, side_2))
        norm = lambda side : math.sqrt(sum([a**2 for a in side]))
        angle = math.acos(dot_product / (norm(side_1) * norm(side_2)))
        
        return 0.5 * norm(side_1) * norm(side_2) * math.sin(angle)

    def flattening(self):
        # Generate all the information regarding each shape.
        for i, face in enumerate(self.faces):
            self.shapes[i + 1] = {}
            self.shapes[i + 1]['original'] = [self.get_pt(face[0]), self.get_pt(face[1]), elf.get_pt(face[2])]
            self.shapes[i + 1]['converted'] = self.to_2d_trig(i + 1)
        
        # For the cheeks, we will define a center point from which the triangles will around.
        CENTER_PT = (0.5*float(PAPER_WIDTH.replace('cm', ''))*cm, 0.5*float(PAPER_HEIGHT.replace('cm', ''))*cm) 

        # Do cheek region 1
        dwg_cheek_1 = svgwrite.Drawing('mask_cheek_1.svg', size=PAPER_SIZE)
        #first_pt = CENTER_PT
        #second_pt = CENTER_PT + ()
        #third_pt = 
        #points = [CENTER_PT, ]
        trig = dwg.polygon(points, stroke=1)
        dwg_cheek_1.add(trig)
        dwg_cheek_1.save()

        # Do cheek region 2
        dwg_cheek_2 = svgwrite.Drawing('mask_cheek_2.svg', size=PAPER_SIZE)
        dwg_cheek_2.save()

        # Do nose region
        dwg_nose = svgwrite.Drawing('mask_nose.svg', size=PAPER_SIZE)
        dwg_nose.save()