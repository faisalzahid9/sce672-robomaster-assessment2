import cv2
import numpy as np

TAG_EDGE_M = 0.08  # Black marker border: 8 cm on our 10 cm grid

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)
detector = cv2.aruco.ArucoDetector(dictionary)

half = TAG_EDGE_M / 2
tag_corners_3d = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)


def detect_target(bgr_image, camera_matrix, distortion, target_id):
    """Return (bearing_rad, camera_forward_m), or None if the chosen tag is absent."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    D = np.asarray(distortion, dtype=np.float64)

    for index, tag_id in enumerate(ids.flatten()):
        if int(tag_id) != target_id:
            continue

        image_corners = corners[index].reshape(4, 2).astype(np.float64)
        centre_x = image_corners[:, 0].mean()
        bearing_rad = float(np.arctan2(centre_x - K[0, 2], K[0, 0]))

        found_pose, _, translation = cv2.solvePnP(
            tag_corners_3d, image_corners, K, D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not found_pose or translation[2, 0] <= 0:
            return None

        return bearing_rad, float(translation[2, 0])

    return None
