import numpy as np
import cv2
from apriltag import apriltag

img_path = './apriltag15.png'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

tag_size = 1
K = np.eye(3)
detector = apriltag("tag36h11")
detections = detector.detect(gray)
print("[INFO] {} total AprilTags detected".format(len(detections)))

print(detections)

for det in detections:
    imagePoints = det['lb-rb-rt-lt'].reshape(1,4,2)  

    ob_pt1 = [-tag_size/2, -tag_size/2, 0.0]
    ob_pt2 = [ tag_size/2, -tag_size/2, 0.0]
    ob_pt3 = [ tag_size/2,  tag_size/2, 0.0]
    ob_pt4 = [-tag_size/2,  tag_size/2, 0.0]
    ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
    object_pts = np.array(ob_pts).reshape(4,3)

    opoints = np.array([
        -1, -1, 0,
        1, -1, 0,
        1,  1, 0,
        -1,  1, 0,
        -1, -1, -2*1,
        1, -1, -2*1,
        1,  1, -2*1,
        -1,  1, -2*1,
    ]).reshape(-1, 1, 3) * 0.5*tag_size
        

    # mtx - the camera calibration's intrinsics
    _, rvec, tvec = cv2.solvePnP(object_pts, imagePoints, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    rvec = np.array([rvec[0,0], rvec[1,0], rvec[2,0]])
    tvec = np.array([tvec[0,0], tvec[1,0], tvec[2,0]])
    rotMat,_ = cv2.Rodrigues(rvec)
    print('tvec  : \n', tvec)
    print('rotMat: \n', rotMat)