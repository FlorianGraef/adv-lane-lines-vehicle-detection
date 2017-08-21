import numpy as np
import cv2
import glob
import pickle

# prepare object points
object_points =  np.zeros((6*9, 3), np.float32)
object_points[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# lists for object and image points
object_points3d = []
image_points = []

# get calibration images
calibration_images = glob.glob('./camera_cal/calibration*.jpg')

for index, imgfile in enumerate(calibration_images):
    img = cv2.imread(imgfile)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(grey, (9, 6), None)

    if ret==True:
        print('Working on ', imgfile)
        object_points3d.append(object_points)
        image_points.append(corners)

        #draw corners and save images
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        out_file_name = './camera_cal/chessboard' + str(index) + '.jpg'
        cv2.imwrite(out_file_name, img)

# load reference image
ref_image = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (ref_image.shape[1], ref_image.shape[0])

# camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points3d, image_points, img_size, None, None)

# save to pickle file
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('./camera_cal/calibration.p', 'wb'))

