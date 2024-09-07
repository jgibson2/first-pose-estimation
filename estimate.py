import cv2
import numpy as np
from apriltag import apriltag
import poselib

if __name__ == "__main__":
    imagepath = 'data/Amp_85in.jpg'
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")
    
    detections = detector.detect(image)
    det = detections[0]
    image_points = np.vstack((det["center"][None, :], det['lb-rb-rt-lt']))
    TAG_HALF_WIDTH = 0.08255 # half of 6.5 in in meters
    world_points = np.array([
        [0.0, 0.0, 0.0],
        [-TAG_HALF_WIDTH, -TAG_HALF_WIDTH, 0.0],
        [TAG_HALF_WIDTH, -TAG_HALF_WIDTH, 0.0],
        [TAG_HALF_WIDTH, TAG_HALF_WIDTH, 0.0],
        [-TAG_HALF_WIDTH, TAG_HALF_WIDTH, 0.0],
    ])
    camera = {'model': 'SIMPLE_PINHOLE', 'width': 1280, 'height': 720, 'params': [1146.45, 640.0, 360.0]}

    pose, info = poselib.estimate_absolute_pose(image_points, world_points, camera, {'max_reproj_error': 16.0}, {})
    print(pose, info)