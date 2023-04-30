import numpy as np
import cv2 as cv

def select_img_from_video(input_file, board_pattern, auto=False, select_all=False, wait_msec=10):
    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file
    f = 0

    # Select images
    img_select = []
    while True:
        f += 1
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow('Camera Calibration', display)
            if auto:
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow('Camera Calibration', display)
                if f % 30 == 0:
                    img_select.append(img)
            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == 27:                  # 'ESC' key: Exit (Complete image selection)
                break
            elif key == ord(' '):          # 'Space' key: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow('Camera Calibration', display)
                key = cv.waitKey()
                if key == 27: # ESC
                    break
                elif key == ord('\r'):
                    img_select.append(img) # 'Enter' key: Select the image

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be 'np.float32'

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)


def pose_estimation_chessboard(input_file, K, dist_coeff, board_pattern = (8, 6), board_cellsize = 0.024):
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    # Prepare a 3D box for simple AR

    h = 5 
    Sx =  2
    Mx = -4
    h = -(h - 1)

    # O word 
    box_lower1 = board_cellsize * (np.array([[2, 2, 0], [2.5, 2, 0], [2.5, 4, 0], [2, 4, 0]]) + np.array([Sx, 0, 0]))
    box_upper1 = board_cellsize * (np.array([[2, 2, -1], [2.5, 2, -1], [2.5, 4, -1], [2, 4, -1]]) + np.array([Sx, 0, h]))

    box_lower2 = board_cellsize * (np.array([[3, 2, 0], [3.5, 2, 0], [3.5, 4, 0], [3, 4, 0]]) + np.array([Sx, 0, 0]))
    box_upper2 = board_cellsize * (np.array([[3, 2, -1], [3.5, 2, -1], [3.5, 4, -1], [3, 4, -1]]) + np.array([Sx, 0, h]))

    box_lower3 = board_cellsize * (np.array([[2.5, 2, 0], [3, 2, 0], [3, 4, 0], [2.5, 4, 0]]) + np.array([Sx, 0, 0]))
    box_upper3 = board_cellsize * (np.array([[2.5, 2, -1], [3, 2, -1], [3, 4, -1], [2.5, 4, -1]]) + np.array([Sx, 0, h]))
    
    box_lower4 = board_cellsize * (np.array([[2.5, 2.25, 0], [3, 2.25, 0], [3, 3.75, 0], [2.5, 3.75, 0]]) + np.array([Sx, 0, 0]))
    box_upper4 = board_cellsize * (np.array([[2.5, 2.25, -1], [3, 2.25, -1], [3, 3.75, -1], [2.5, 3.75, -1]]) + np.array([Sx, 0, h]))
    

    box_lower = np.concatenate((box_lower1, box_lower2), axis=0)

    # H word 
    box_lower5 = board_cellsize * (np.array([[6, 2, 0], [6.5, 2, 0], [6.5, 4, 0], [6, 4, 0]]) + np.array([Mx, 0, 0]))
    box_upper5 = board_cellsize * (np.array([[6, 2, -1], [6.5, 2, -1], [6.5, 4, -1], [6, 4, -1]]) + np.array([Mx, 0, h]))

    box_lower6 = board_cellsize * (np.array([[7, 2, 0], [7.5, 2, 0], [7.5, 4, 0], [7, 4, 0]]) + np.array([Mx, 0, 0]))
    box_upper6 = board_cellsize * (np.array([[7, 2, -1], [7.5, 2, -1], [7.5, 4, -1], [7, 4, -1]]) + np.array([Mx, 0, h]))

    box_lower7 = board_cellsize * (np.array([[6.5, 2.75, 0], [7, 2.75, 0], [7, 3.25, 0], [6.5, 3.25, 0]]) + np.array([Mx, 0, 0]))
    box_upper7 = board_cellsize * (np.array([[6.5, 2.75, -1], [7, 2.75, -1], [7, 3.25, -1], [6.5, 3.25, -1]]) + np.array([Mx, 0, h]))

    box_lower = np.concatenate((box_lower, box_lower4), axis=0)


    # Prepare 3D points on a chessboard
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Estimate the camera pose
        complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the box on the image
            line_lower1, _ = cv.projectPoints(box_lower1, rvec, tvec, K, dist_coeff)
            line_upper1, _ = cv.projectPoints(box_upper1, rvec, tvec, K, dist_coeff)
            line_lower2, _ = cv.projectPoints(box_lower2, rvec, tvec, K, dist_coeff)
            line_upper2, _ = cv.projectPoints(box_upper2, rvec, tvec, K, dist_coeff)
            line_lower3, _ = cv.projectPoints(box_lower3, rvec, tvec, K, dist_coeff)
            line_upper3, _ = cv.projectPoints(box_upper3, rvec, tvec, K, dist_coeff)
            line_lower4, _ = cv.projectPoints(box_lower4, rvec, tvec, K, dist_coeff)
            line_upper4, _ = cv.projectPoints(box_upper4, rvec, tvec, K, dist_coeff)
            line_lower5, _ = cv.projectPoints(box_lower5, rvec, tvec, K, dist_coeff)
            line_upper5, _ = cv.projectPoints(box_upper5, rvec, tvec, K, dist_coeff)
            line_lower6, _ = cv.projectPoints(box_lower6, rvec, tvec, K, dist_coeff)
            line_upper6, _ = cv.projectPoints(box_upper6, rvec, tvec, K, dist_coeff)
            line_lower7, _ = cv.projectPoints(box_lower7, rvec, tvec, K, dist_coeff)
            line_upper7, _ = cv.projectPoints(box_upper7, rvec, tvec, K, dist_coeff)
            
            cv.polylines(img, [np.int32(line_lower1)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_lower2)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_lower3)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_lower4)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_lower5)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_lower6)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_lower7)], True, (255, 0, 0), 2)


            for b, t in zip(line_lower1, line_upper1):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower2, line_upper2):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower3, line_upper3):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower4, line_upper4):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower5, line_upper5):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower6, line_upper6):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower7, line_lower7):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 0, 255), 2)

            # Print the camera position
            R, _ = cv.Rodrigues(rvec)
            cv.polylines(img, [np.int32(line_upper1)], True, (0, 0, 255), 2)
            cv.polylines(img, [np.int32(line_upper2)], True, (0, 0, 255), 2)
            cv.polylines(img, [np.int32(line_upper3)], True, (0, 0, 255), 2)
            cv.polylines(img, [np.int32(line_upper4)], True, (0, 0, 255), 2)
            cv.polylines(img, [np.int32(line_upper5)], True, (0, 0, 255), 2)
            cv.polylines(img, [np.int32(line_upper6)], True, (0, 0, 255), 2)
            cv.polylines(img, [np.int32(line_upper7)], True, (0, 0, 255), 2)

            # Print the camera position
            R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Show the image and process the key event
        cv.imshow('Pose Estimation (Chessboard)', img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    input_file = './data/chess.mp4'
    auto_calibration = True
    board_pattern = (8, 6)
    board_cellsize = 0.024

    img_select = select_img_from_video(input_file, board_pattern, auto_calibration)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    pose_estimation_chessboard(input_file, K, dist_coeff.flatten())