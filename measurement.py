import copy
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
        
        # For the regions to be flattened
        self.nose_eye_region = {}
        self.cheek_1_region = {}
        self.cheek_2_region = {}

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
            if line[0] != '#':
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
            return math.sqrt((pt_1[1] - pt_0[1])**2 + (pt_1[0] - pt_0[0])**2)  
        else:
            return -1

    def measure_ratio(self):
        # Get the distance between the corner of the mouth and the tip of the nose.
        tip_nose_pt = self.get_pt(30)
        corner_mouth_pt = self.get_pt(49)
        dist_3d_space = self.distance(tip_nose_pt, corner_mouth_pt)
        
        if self.units == 'in':
            return self.ref_dist * 2.54 / dist_3d_space
        else:
            return self.ref_dist / dist_3d_space

    # We want to transform the 3D coordinates making up the triangle into 2D coordinates.
    def to_2d_trig(self, face_id):
        v_1 = self.shapes[face_id]['original_3d_px'][0]
        v_2 = self.shapes[face_id]['original_3d_px'][1]
        v_3 = self.shapes[face_id]['original_3d_px'][2]

        h = math.sqrt((v_2[2] - v_1[2])**2 + (v_2[1] - v_1[1])**2 + (v_2[0] - v_1[0])**2)
        i = ((v_3[2] - v_1[2])*(v_2[2] - v_1[2]) + (v_3[1] - v_1[1])*(v_2[1] - v_1[1]) + (v_3[0] - v_1[0])*(v_2[0] - v_1[0]))/h
        j = math.sqrt((v_3[2] - v_1[2])**2 + (v_3[1] - v_1[1])**2 + (v_3[0] - v_1[0])**2 - i**2)

        return [[0, 0], [h, 0], [i, j]]

    def calculate_area(self, vertices):
        if (len(vertices[0]) != len(vertices[1]) and len(vertices[0]) != len(vertices[2])):
            return -1
        
        side_1 = [val_2 - val_1 for (val_2, val_1) in zip(vertices[1], vertices[0])]
        side_2 = [val_2 - val_1 for (val_2, val_1) in zip(vertices[2], vertices[0])]
        
        norm = lambda side : math.sqrt(sum([a**2 for a in side]))
        angle = self.calculate_angle(side_1, side_2)
        
        return 0.5 * norm(side_1) * norm(side_2) * math.sin(angle)

    def calculate_angle(self, vector_1, vector_2):
        dot_product = sum(val_2 * val_1 for (val_2, val_1) in zip(vector_1, vector_2))
        norm = lambda side : math.sqrt(sum([a**2 for a in side]))
        angle = math.acos(dot_product / (norm(vector_1) * norm(vector_2)))
        return angle
    
    def get_vec(self, arr_1, arr_2):
        vec = [arr_2[0] - arr_1[0], arr_2[1] - arr_1[1]]
        return vec

    def alignment_nose_eye(self, align_wrt, to_be_corrected, connect_pt):
        side_match_1 = (-1, -1)
        side_match_2 = (-1, -1)
        for combo_1 in combinations([0, 1, 2], 2):
            for combo_2 in combinations([0, 1, 2], 2):
                dist_1 = self.distance(self.nose_eye_region[align_wrt][combo_1[0]], self.nose_eye_region[align_wrt][combo_1[1]])
                dist_2 = self.distance(self.nose_eye_region[to_be_corrected][combo_2[0]], self.nose_eye_region[to_be_corrected][combo_2[1]])
                if abs(dist_1 - dist_2) < 0.01:
                    side_match_1 = combo_1 
                    side_match_2 = combo_2

        angle_to_rotate = self.calculate_angle(self.get_vec(self.nose_eye_region[align_wrt][side_match_1[0]], self.nose_eye_region[align_wrt][side_match_1[1]]), 
        self.get_vec(self.nose_eye_region[to_be_corrected][side_match_2[0]], self.nose_eye_region[to_be_corrected][side_match_2[1]]))

        for i, _ in enumerate(self.nose_eye_region[to_be_corrected]):
            temp_1 = self.nose_eye_region[align_wrt][connect_pt][0] + math.cos(angle_to_rotate) * self.nose_eye_region[to_be_corrected][i][0] - math.sin(angle_to_rotate) * self.nose_eye_region[to_be_corrected][i][1]  
            temp_2 = self.nose_eye_region[align_wrt][connect_pt][1] + math.sin(angle_to_rotate) * self.nose_eye_region[to_be_corrected][i][0] + math.cos(angle_to_rotate) * self.nose_eye_region[to_be_corrected][i][1]  
            self.nose_eye_region[to_be_corrected][i][0] = temp_1
            self.nose_eye_region[to_be_corrected][i][1] = temp_2

    def alignment_cheek(self, align_wrt, to_be_corrected, connect_pt):
        side_match_1 = (-1, -1)
        side_match_2 = (-1, -1)
        for combo_1 in combinations([0, 1, 2], 2):
            for combo_2 in combinations([0, 1, 2], 2):
                dist_1 = self.distance(self.cheek_1_region[align_wrt][combo_1[0]], self.cheek_1_region[align_wrt][combo_1[1]])
                dist_2 = self.distance(self.cheek_1_region[to_be_corrected][combo_2[0]], self.cheek_1_region[to_be_corrected][combo_2[1]])
                if abs(dist_1 - dist_2) < 0.01:
                    side_match_1 = combo_1 
                    side_match_2 = combo_2

        angle_to_rotate = self.calculate_angle(self.get_vec(self.cheek_1_region[align_wrt][side_match_1[0]], self.cheek_1_region[align_wrt][side_match_1[1]]), 
        self.get_vec(self.cheek_1_region[to_be_corrected][side_match_2[0]], self.cheek_1_region[to_be_corrected][side_match_2[1]]))
        
        for i, _ in enumerate(self.cheek_1_region[to_be_corrected]):
            temp_1 = self.cheek_1_region[align_wrt][connect_pt][0] + math.cos(angle_to_rotate) * self.cheek_1_region[to_be_corrected][i][0] - math.sin(angle_to_rotate) * self.cheek_1_region[to_be_corrected][i][1]  
            temp_2 = self.cheek_1_region[align_wrt][connect_pt][1] + math.sin(angle_to_rotate) * self.cheek_1_region[to_be_corrected][i][0] + math.cos(angle_to_rotate) * self.cheek_1_region[to_be_corrected][i][1]  
            self.cheek_1_region[to_be_corrected][i][0] = temp_1
            self.cheek_1_region[to_be_corrected][i][1] = temp_2

    def symmetry(self):
        return

    def flattening(self, show_info=False):
        # For the cheeks, we will define a center point from which the triangles will around.
        CENTER_PT = (0.5*float(PAPER_WIDTH.replace('cm', ''))*cm, 0.5*float(PAPER_HEIGHT.replace('cm', ''))*cm) 

        # Do cheek region 1
        dwg_mask = svgwrite.Drawing('./svg/mask.svg', size=PAPER_SIZE)

        # Generate all the information regarding each shape.
        for i, face in enumerate(self.faces):
            self.shapes[i + 1] = {}
            self.shapes[i + 1]['original_3d_px'] = [self.get_pt(face[0]), self.get_pt(face[1]), self.get_pt(face[2])]
            self.shapes[i + 1]['converted_2d_px'] = self.to_2d_trig(i + 1)
            self.shapes[i + 1]['converted_2d_cm'] = []
            
            for pt in self.shapes[i + 1]['converted_2d_px']:
                converted_pt = [self.ratio * pt[0], self.ratio * pt[1]]
                self.shapes[i + 1]['converted_2d_cm'].append(converted_pt)

            if show_info:
                original_area = self.calculate_area(self.shapes[i + 1]['original_3d_px'])
                converted_area = self.calculate_area(self.shapes[i + 1]['converted_2d_px'])
                diff_area = abs(converted_area - original_area)
                print('Original area (px): ' + str(original_area))
                print('Converted area (px): ' + str(converted_area))
                print('Difference (px): ' + str(diff_area) + '\n')

        # Cheek 1 flatten region
        self.cheek_1_region[1] = self.shapes[7]['converted_2d_cm']

        for i in range(2, 10):
            self.cheek_1_region[i] = self.shapes[i + 6]['converted_2d_cm']
            self.cheek_1_region[i][2][1] = -1*self.cheek_1_region[i][2][1]

            for j, _ in enumerate(self.cheek_1_region[i]):
                self.cheek_1_region[i][j][0] = math.cos(math.pi) * self.cheek_1_region[i][j][0] - math.sin(math.pi) * self.cheek_1_region[i][j][1]  
                self.cheek_1_region[i][j][1] = math.sin(math.pi) * self.cheek_1_region[i][j][0] + math.cos(math.pi) * self.cheek_1_region[i][j][1]  
            
            self.alignment_cheek(i - 1, i, 0)
            shift_pt = [-1 * val for val in self.get_vec(self.cheek_1_region[i - 1][0], self.cheek_1_region[i][1])]
            
            for j, _ in enumerate(self.cheek_1_region[i]):
                self.cheek_1_region[i][j][0] = self.cheek_1_region[i][j][0] + shift_pt[0]
                self.cheek_1_region[i][j][1] = self.cheek_1_region[i][j][1] + shift_pt[1]
    
        # For the first cheek region
        self.alignment_cheek(2, 1, 0)
        shift_pt = [-1 * val for val in self.get_vec(self.cheek_1_region[2][2], self.cheek_1_region[1][1])]
        
        for j, _ in enumerate(self.cheek_1_region[1]):
            self.cheek_1_region[1][j][0] = self.cheek_1_region[1][j][0] + shift_pt[0]
            self.cheek_1_region[1][j][1] = self.cheek_1_region[1][j][1] + shift_pt[1]

        # For the last cheek triangle
        self.cheek_1_region[10] = self.shapes[16]['converted_2d_cm']
        
        for j, _ in enumerate(self.cheek_1_region[10]):
            self.cheek_1_region[10][j][0] = math.cos(math.pi) * self.cheek_1_region[10][j][0] - math.sin(math.pi) * self.cheek_1_region[10][j][1]  
            self.cheek_1_region[10][j][1] = math.sin(math.pi) * self.cheek_1_region[10][j][0] + math.cos(math.pi) * self.cheek_1_region[10][j][1]  
        
        self.alignment_cheek(9, 10, 0)

        # Make cheek straight and vertical
        base_vec = self.get_vec(self.cheek_1_region[10][0], self.cheek_1_region[10][1])
        if base_vec[0] < 0:
            base_angle = -1*self.calculate_angle(base_vec, [-1, 0])
        else:
            base_angle = self.calculate_angle(base_vec, [1, 0])

        # Perform transformations on the entire mask
        for i, _ in enumerate(self.cheek_1_region):
            for j, _ in enumerate(self.cheek_1_region[i + 1]):
                # Rotate all the points by the base angle
                self.cheek_1_region[i + 1][j][0] = math.cos(base_angle) * self.cheek_1_region[i + 1][j][0] - math.sin(base_angle) * self.cheek_1_region[i + 1][j][1]  
                self.cheek_1_region[i + 1][j][1] = math.sin(base_angle) * self.cheek_1_region[i + 1][j][0] + math.cos(base_angle) * self.cheek_1_region[i + 1][j][1]  
                # Shift y-position of mask
                self.cheek_1_region[i + 1][j][1] = self.cheek_1_region[i + 1][j][1] + 7

        # Figure out how far to shift the mask in the x-position such that it is 1 cm away from the border. Also figure out what is the maximum y
        lowest_x = lowest_y = 1
        highest_x = highest_y = -1
        for i, _ in enumerate(self.cheek_1_region):
            for j, _ in enumerate(self.cheek_1_region[i + 1]):
                x_val = self.cheek_1_region[i + 1][j][0]
                y_val = self.cheek_1_region[i + 1][j][1]
                if x_val <= lowest_x:
                    lowest_x = x_val
                elif x_val >= highest_x:
                    highest_x = x_val 
                if y_val <= lowest_y:
                    lowest_y = y_val
                elif y_val >= highest_y:
                    highest_y = y_val
        
        shift = 0
        if lowest_x < 0:
            shift = abs(lowest_x) + 1
        if highest_x + shift > 21.59:
            raise Exception('CANNOT FIT ON THE PAGE')
        for i, _ in enumerate(self.cheek_1_region):
            for j, _ in enumerate(self.cheek_1_region[i + 1]):
                 self.cheek_1_region[i + 1][j][0] =  self.cheek_1_region[i + 1][j][0] + shift

        for i in range(1, 11):
            for comb in combinations([0, 1, 2], 2):
                start_tuple_1 = (self.cheek_1_region[i][comb[0]][0]*cm, self.cheek_1_region[i][comb[0]][1]*cm)
                end_tuple_1 = (self.cheek_1_region[i][comb[1]][0]*cm, self.cheek_1_region[i][comb[1]][1]*cm)
                dwg_mask.add(dwg_mask.line(start=start_tuple_1, end=end_tuple_1, stroke='green'))

        # Nose and eye flatten region
        self.nose_eye_region[1] = self.shapes[1]['converted_2d_cm']
        
        self.nose_eye_region[2] = self.shapes[2]['converted_2d_cm']
        self.alignment_nose_eye(1, 2, 1)

        self.nose_eye_region[3] = self.shapes[3]['converted_2d_cm']
        self.alignment_nose_eye(2, 3, 0)

        self.nose_eye_region[4] = copy.deepcopy(self.shapes[1]['converted_2d_cm'])
        self.nose_eye_region[4][2][1] = -1*self.nose_eye_region[4][2][1]

        self.nose_eye_region[5] = copy.deepcopy(self.shapes[2]['converted_2d_cm'])
        self.nose_eye_region[5][1][1] = -1*self.nose_eye_region[5][1][1]
        self.nose_eye_region[5][2][1] = -1*self.nose_eye_region[5][2][1]

        self.nose_eye_region[6] = copy.deepcopy(self.shapes[3]['converted_2d_cm'])
        self.nose_eye_region[6][1][1] = -1*self.nose_eye_region[6][1][1]
        self.nose_eye_region[6][2][1] = -1*self.nose_eye_region[6][2][1]

        shift = highest_x + shift + 2
        for i, _ in enumerate(self.nose_eye_region):
            for j, _ in enumerate(self.nose_eye_region[i + 1]):
                self.nose_eye_region[i + 1][j][1] =  self.nose_eye_region[i + 1][j][1] + 14
                self.nose_eye_region[i + 1][j][0] =  self.nose_eye_region[i + 1][j][0] + shift

        for i in range(1, 7):
            for comb in combinations([0, 1, 2], 2):
                start_tuple = (self.nose_eye_region[i][comb[0]][0]*cm, self.nose_eye_region[i][comb[0]][1]*cm)
                end_tuple = (self.nose_eye_region[i][comb[1]][0]*cm, self.nose_eye_region[i][comb[1]][1]*cm)
                dwg_mask.add(dwg_mask.line(start=start_tuple, end=end_tuple, stroke='green'))

        dwg_mask.save()