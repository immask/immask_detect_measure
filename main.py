import pdb; pdb.set_trace()

from calibrate import Calibrate
import cv2
from detector import Detector
import glob
from measurement import Measurement
import numpy as np
import pickle

def main():
    # Get some calibration information that is dependent on the phone model used to take a 
    # picture
    #device = 'iPhone_8'
    #find_calibration_param = glob.glob('./camera_correction/*' + device + '*')
    
    #if not find_calibration_param:
    #    return
        
    #calibration_param = pickle.load(open(find_calibration_param[0], 'rb'))
    #mtx = calibration_param['mtx']
    #dist = calibration_param['dist']

    # Getting the image
    img = cv2.imread('./img/andy_selfie.jpg')
    
    # Here, we will be correcting the image in question using the calibration information and
    # generating a new camera matrix
    #h, w = img.shape[:2]
    
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #img_to_show = np.hstack((img, dst))

    # Showing the outcome
    #cv2.imshow('Undistorted', img_to_show)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    d = Detector.load_image(img)
    d.detect_3d()

    # Instance of measurement for getting the dimensions of the mask.
    meas = Measurement(features_3d=d.features_3d, ref_dist=5.5, units='cm')
    meas.find_all_pts()
    meas.find_all_faces()
    meas.save_stl()
    meas.preview()
    meas.flattening()
    
if __name__ == "__main__":
	main()