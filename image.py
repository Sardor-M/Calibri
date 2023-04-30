# import numpy as np
# import cv2
# import glob

# # Define the chessboard size and square size
# chessboard_size = (7, 6)
# square_size = 0.02423  # meters

# # Define the object points (3D coordinates of the chessboard corners)
# object_points = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
# object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
# object_points *= square_size

# # Create an array to store the object points and image points for each image
# object_points_array = []
# image_points_array = []

# # Capture images of the chessboard
# images = glob.glob('chessboard/*.jpg')

# # Load the camera calibration parameters
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])

# # Create a SM object using the camera calibration parameters
# sm_object = cv2.omnidir.OmniCameraModel(camera_matrix, dist_coeffs, xi)

# # Iterate through each image and detect the chessboard corners
# for image_file in images:
#     image = cv2.imread(image_file)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

#     if ret:
#         # Refine the corner coordinates to subpixel accuracy
#         corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

#         # Add the object points and image points to the arrays
#         object_points_array.append(object_points)
#         image_points_array.append(corners)

#         # Draw the chessboard corners on the image
#         cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

#         # Project the "OK" words in AR using the camera calibration parameters
#         rvec, tvec = cv2.omnidir.solvePnP(np.array([[0, 0, 0], [0, 0, -1]]), corners, camera_matrix, dist_coeffs, xi)
#         img_points, _ = cv2.omnidir.projectPoints(np.array([[0, 0, 0], [0, 0, -1]]), rvec, tvec, camera_matrix, dist_coeffs, xi)
#         img_points = np.int32(img_points).reshape(-1, 2)
#         cv2.putText(image, "OK", tuple(img_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, omnidirectional=True)

#     # Display the image
#     cv2.imshow('AR Chessboard', image)
#     cv2.waitKey(1000)

# # Calibrate the camera using the object points and image points
# ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_array, image_points_array, gray.shape[::-1], None, None)

# cv2.destroyAllWindows()
