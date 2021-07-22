# 这个是验证外参矩阵的程序，这里面会保存在matlab中计算好的四个外参矩阵
# 并且投影两个标定块的四个角（共计8个）验证外参矩阵的正确性，因为标定的时候只用到了3个点

import cv2 as cv
import numpy as np

# ====================这里是常参数定义区==========================
K = np.array([[238.87052313877663, 0.0, 328.1263268763279],
              [0.0, 236.36510731401265, 228.35450912275593],
              [0.0, 0.0, 1.0]])
D = np.array([-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566])
DIM = (640, 480)
# ====================四个块在世界坐标系下=======================
RT_front = np.array([[1., 0, 0, 0],
                    [0.0426, -0.5315, 0, 292.3681],
                    [0.0076, 0.5601, 0, 14.0965]])


RT_front1 = np.array([[-0.0086, -0.0005, 0, -0.0046],
                    [-0.0002, 0.0044, 0,  -2.4795],
                    [-0.0005, -0.0078, 0, 1]])

RT_front2 = np.array([[-0.9867,    0.0401,    0.0507,    0.6864],
   [-0.0178,    0.4885,   -0.8751, -271.1025],
   [-0.0716,   -0.8839,   -0.4813,  113.5791]])


# RT_right = np.array([[0, -1, 0, 0],
#                     [-0.6849, -0.0145, 0, 259.9521],
#                     [0.7588, -0.0651, 0, -16.1225]])

RT_right = np.array([[0.0007, 0.0296, 0, -0.0829],
                    [0.0200, 0.0004, 0, -7.5936],
                    [-0.0245, 0.0016, 0, 1]])

RT_back = np.array([[0.0089, -0.0002, 0, -0.1399],
                    [-0.0002, -0.0048, 0, -2.3733],
                    [0.0001, 0.0081, 0, 1]])


RT_left = np.array([[-0.000, -0.0234, 0, 0.2424],
                    [-0.0149, -0.0007, 0, -6.0948],
                    [0.0204, -0.0020, 0, 1]])


# 给定一个标定块的中心点和相机外参，直接返回四个标定块的四角在相机坐标系下的坐标点
def getfourcorners(center, extric_mtx):
    cameraCornersArray = np.zeros([4, 1, 3], np.float)
    # 左上角
    cornerArray = [center[0] - 70, center[1] + 70, 0, 1]
    cornerArray = np.dot(extric_mtx, cornerArray)
    cameraCornersArray[0][0] = cornerArray
    print('test:', cameraCornersArray[0][0][0]/cameraCornersArray[0][0][2], cameraCornersArray[0][0][1]/cameraCornersArray[0][0][2])
    # 右上角
    cornerArray = [center[0] + 70, center[1] + 70, 0, 1]
    cornerArray = np.dot(extric_mtx, cornerArray)
    cameraCornersArray[1][0] = cornerArray
    # 左下角
    cornerArray = [center[0] - 70, center[1] - 70, 0, 1]
    cornerArray = np.dot(extric_mtx, cornerArray)
    cameraCornersArray[2][0] = cornerArray
    # 右下角
    cornerArray = [center[0] + 70, center[1] - 70, 0, 1]
    cornerArray = np.dot(extric_mtx, cornerArray)
    cameraCornersArray[3][0] = cornerArray
    # print('four corner is:', cameraCornersArray)
    return cameraCornersArray

# # 映射坐标，把世界坐标系的点映射到图中，输入参数为规定形式的点数组和相机外参矩阵
# def projectWorldPoint(conerArray, extric_mtx):


# 这个函数是微调两个摄像头的数据，对两个摄像头联合标定
_front = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/0_front.jpg", 1)
_right = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/0_right.jpg", 1)
_back = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/0_back.jpg", 1)
_left = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/0_left.jpg", 1)
rvecs = np.zeros((3, 1), dtype=np.float64)
tvecs = np.zeros((3, 1), dtype=np.float64)
# 前方摄像头
cameraCorners = getfourcorners((300, 300), RT_front)
undistortedCorners, jacobian = cv.fisheye.projectPoints(cameraCorners, rvecs, tvecs, K, D)
for i in range(4):
    uv = (int(undistortedCorners[i][0][0]), int(undistortedCorners[i][0][1]))
    cv.circle(_front, uv, 3, (255, 0, 0), -1)

cameraCorners = getfourcorners((300, 300), RT_front2)
undistortedCorners, jacobian = cv.fisheye.projectPoints(cameraCorners, rvecs, tvecs, K, D)
print(undistortedCorners)
for i in range(4):
    uv = (int(undistortedCorners[i][0][0]), int(undistortedCorners[i][0][1]))
    cv.circle(_front, uv, 3, (255, 0, 255), -1)
cv.imshow('Front', _front)
cv.waitKey(0)
