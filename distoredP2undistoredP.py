# 这个程序的作用是把拍摄的原图进行点击，将相应点投影到去畸变的图上

import cv2 as cv
import numpy as np

K = np.array([[238.87052313877663, 0.0, 328.1263268763279],
              [0.0, 236.36510731401265, 228.35450912275593],
              [0.0, 0.0, 1.0]])
D = np.array([-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566])
DIM = (640, 480)
rvecs = np.zeros((3, 1), dtype=np.float64)
tvecs = np.zeros((3, 1), dtype=np.float64)

undistort_x = 0
undistort_y = 0


def mousecallback(event, x, y, flags, param):
    global undistort_x, undistort_y
    if event == cv.EVENT_LBUTTONUP:
        undistort_x, undistort_y, _ = caculate_undistorted([x, y])
        print('[', x, ',', y, '], to [', undistort_x, undistort_y, ']')


def caculate_undistorted(uv):
    uvPoint = np.zeros([1, 1, 2], np.float)
    uvPoint[0][0] = uv
    out = cv.fisheye.undistortPoints(uvPoint, K, D)
    # 利用相等的x坐标去求真实的xyz坐标
    xy = np.zeros([3, 1])
    xy[0] = out[0][0][0]
    xy[1] = out[0][0][1]
    xy[2] = 1
    uv = np.dot(K, xy)
    return uv


if __name__ == "__main__":
    img = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/1_front.jpg", 1)
    shaped = img.shape
    row_extend = 600
    col_extend = 800
    map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.identity(3), K, (shaped[1], shaped[0]), cv.CV_32FC1)
    dst = cv.remap(img, map1, map2, cv.INTER_CUBIC)
    sheet = np.zeros([row_extend+shaped[0], col_extend+shaped[1], 3], np.uint8)
    sheet[0:shaped[0], col_extend//2:col_extend//2+shaped[1]] = dst
    while True:
        cv.circle(sheet, (undistort_x+col_extend//2, undistort_y), 4, (0, 255, 255), 3)
        cv.imshow('distort', img)
        cv.imshow('undistorted', sheet)
        cv.setMouseCallback('distort', mousecallback)
        cv.waitKey(100)