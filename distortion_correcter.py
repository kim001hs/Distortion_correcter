import cv2 as cv
import numpy as np

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    # 비디오 열기
    video = cv.VideoCapture(video_file)
    if not video.isOpened():
        print("Error opening video file")
        return []

    img_select = []
    while True:
        valid, img = video.read()
        if not valid:
            break

        print("Frame shape:", img.shape)  # 디버깅용

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, corners = cv.findChessboardCorners(gray, board_pattern)

        vis = img.copy()
        cv.drawChessboardCorners(vis, board_pattern, corners, complete)

        cv.namedWindow('Select chessboard images', cv.WINDOW_NORMAL)
        cv.resizeWindow('Select chessboard images', 1280, 720)
        cv.imshow('Select chessboard images', vis)
        
        key = cv.waitKey(wait_msec)

        if complete and (select_all or key == ord(' ')):  # 스페이스바를 누르면 이미지 선택
            img_select.append(img.copy())
            print(f"Selected image #{len(img_select)}")

        if key == 27:  # ESC 키로 종료
            break

    video.release()
    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            # 코너 정밀화
            pts = cv.cornerSubPix(
                gray, pts, (11, 11), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            img_points.append(pts)

    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    # 체스보드의 3D 점 생성
    obj_pts = np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])], dtype=np.float32)
    obj_points = [obj_pts * board_cellsize] * len(img_points)

    # 카메라 보정
    ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags
    )

    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist_coeff)
    return ret, K, dist_coeff, rvecs, tvecs

if __name__ == "__main__":
    board_pattern = (8, 6)
    board_cellsize = 25.0  # mm 단위, 필요시 조정

    video_path = "./data/chessboard.mp4"  # 비디오 파일 경로
    selected_images = select_img_from_video(video_path, board_pattern)

    if selected_images:
        calib_camera_from_chessboard(selected_images, board_pattern, board_cellsize)
    else:
        print("No images selected for calibration.")
