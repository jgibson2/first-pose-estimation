import cv2
import numpy as np
from apriltag import apriltag
import poselib
import json
import timeit
from pathlib import Path
from scipy.spatial.transform import Rotation

TAG_HALF_WIDTH = 0.08255 # half of 6.5 in in meters

if __name__ == "__main__":
    for image_path in sorted(list(Path('data').glob('*.jpg'))):
        print(f'Processing image {image_path.name}')
        transforms_path = Path('data/2024-crescendo.json')
        image = cv2.imread(str(image_path.absolute()), cv2.IMREAD_GRAYSCALE)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        with open(transforms_path, 'r') as tf:
            transforms_data = json.load(tf)

        tags_transform_map = {tag["ID"] : tag for tag in transforms_data["tags"]}
        detector = apriltag("tag36h11")

        t_0 = timeit.default_timer()
        detections = detector.detect(image)
        world_points = []
        image_points = []
        rect_coords = []
        for det in detections:
            if det["id"] not in tags_transform_map:
                print(f'ERROR: Could not find id {det["id"]} in transforms map!')
                continue
            # homogenous coords
            # "field space" is X left, y front, z up
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

            rect_coords.append(np.round(det['lb-rb-rt-lt']).astype(np.int32).reshape(-1, 1, 2))

        if not image_points:
            print('ERROR: No image points detected.')
            continue
        else:
            print(f'Found tags {", ".join([str(det["id"]) for det in detections])}')

        world_points = np.vstack(world_points)
        image_points = np.vstack(image_points)

        # print('World points:\n', world_points)
        # print('Image points:\n', image_points)

        camera = {'model': 'SIMPLE_PINHOLE', 'width': 1280, 'height': 720, 'params': [1146.45, 640.0, 360.0]}

        # world to cam
        pose, info = poselib.estimate_absolute_pose(image_points, world_points, camera, {'max_reproj_error': 12.0}, {})
        print('Pose estimation info: ', info)
        cam_F_world = np.eye(4)
        cam_F_world[:3, :3] = pose.R
        cam_F_world[:3, 3] = pose.t
        world_F_cam = np.linalg.inv(cam_F_world)

        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0) * 10 ** 3, 3)
        print(f"Elapsed time: {elapsed_time} ms")

        cam_origin_in_world = (world_F_cam @ np.array([0.0, 0.0, 0.0, 1.0]).T)[:3]
        print('Camera origin in world coordinate system:', cam_origin_in_world)
        for det in detections:
            if det["id"] in tags_transform_map:
                pose = tags_transform_map[det["id"]]["pose"]
                tag_center = np.array([pose["translation"]["x"], pose["translation"]["y"], pose["translation"]["z"]])
                print(f'Distance to tag {det["id"]} center: ', np.linalg.norm(cam_origin_in_world - tag_center))

        image_bgr = cv2.polylines(image_bgr, rect_coords, True, (0, 255, 0), 4)
        cv2.imshow(image_path.name, image_bgr)
        cv2.waitKey(0)
        cv2.destroyWindow(image_path.name)
        print('=' * 80)

    cv2.destroyAllWindows()
