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


def pose_estimation_chessboard(input_file, K, dist_coeff, board_pattern = (10, 7), board_cellsize = 0.024):
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    # Prepare 3D points on a chessboard
    object_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    input_file = './data/chess.mp4'
    auto = True
    board_pattern = (10, 7)
    board_cellsize = 0.024

    img_select = select_img_from_video(input_file, board_pattern, auto)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    pose_estimation_chessboard(input_file, K, dist_coeff.flatten())