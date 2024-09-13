import cv2
import numpy as np
from apriltag import apriltag
import poselib
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

TAG_HALF_WIDTH = 0.08255 # half of 6.5 in in meters

if __name__ == "__main__":
    image_path = Path('data/BackAmpZone_117in.jpg')
    transforms_path = Path('data/2024-crescendo.json')
    image = cv2.imread(str(image_path.absolute()), cv2.IMREAD_GRAYSCALE)
    with open(transforms_path, 'r') as tf:
        transforms_data = json.load(tf)

    tags_transform_map = {tag["ID"] : tag for tag in transforms_data["tags"]}
    detector = apriltag("tag36h11")
    
    detections = detector.detect(image)
    world_points = []
    image_points = []
    for det in detections:
        if det["id"] not in tags_transform_map:
            print(f'ERROR: Could not find id {det["id"]} in transforms map!')
            continue
        else:
            print(f'Found tag {det["id"]}')
        # homogenous coords
        tag_points = np.array([
            [ 0.0,  0.0,             0.0,            1.0],
            [ 0.0, -TAG_HALF_WIDTH, -TAG_HALF_WIDTH, 1.0],
            [ 0.0,  TAG_HALF_WIDTH, -TAG_HALF_WIDTH, 1.0],
            [ 0.0,  TAG_HALF_WIDTH,  TAG_HALF_WIDTH, 1.0],
            [ 0.0, -TAG_HALF_WIDTH,  TAG_HALF_WIDTH, 1.0],
        ])
        transform = np.eye(4)
        pose = tags_transform_map[det["id"]]["pose"]
        translation = pose["translation"]
        quat = pose["rotation"]["quaternion"]
        transform[:3, :3] = Rotation.from_quat(np.array([quat["X"], quat["Y"], quat["Z"], quat["W"]]), scalar_first=False).as_matrix()
        transform[:3, 3] = np.array([translation["x"], translation["y"], translation["z"]])
        world_points.append(np.einsum('ij,nj->ni', transform, tag_points)[:, :3])
        img_pts = np.vstack([det["center"][None, :], det['lb-rb-rt-lt']])
        image_points.append(img_pts)

    world_points = np.vstack(world_points)
    image_points = np.vstack(image_points)

    print('World points:\n', world_points)
    print('Image points:\n', image_points)

    camera = {'model': 'SIMPLE_PINHOLE', 'width': 1280, 'height': 720, 'params': [1146.45, 640.0, 360.0]}

    # cam_F_world
    pose, info = poselib.estimate_absolute_pose(image_points, world_points, camera, {'max_reproj_error': 16.0}, {})
    print(pose, info)
    # world_F_cam
    print('World origin from camera translation: ', -pose.R.T @ pose.t)
