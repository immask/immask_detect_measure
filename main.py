import pdb; pdb.set_trace()

from calibrate import Calibrate
import cv2
from detector import Detector
import glob
from measurement import Measurement
import numpy as np
import pickle

def main():
    # Getting the image
    img = cv2.imread('./img/andy_selfie.jpg')

    # Performing the detection
    d = Detector.load_image(img)
    d.detect_3d()

    # Instance of measurement for getting the dimensions of the mask.
    meas = Measurement(features_3d=d.features_3d, ref_dist=7.5, units='cm')
    meas.find_all_pts()
    meas.find_all_faces()
    #meas.save_stl()
    #meas.preview()
    meas.flattening()
    
if __name__ == "__main__":
	main()