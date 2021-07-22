import numpy as np
import math
import cv2 as cv
import copy


# brief：使用测量的平移量和旋转量做坐标变换的方法，同时也提供给使用者实时微调测量参数的track——bar


# =-=-=-=-=-=-=-=下面的是通过opencv进行的相机标定数据-=-=-=-=-=-=-=-=-=-=-
K = np.array([[238.87052313877663, 0.0, 328.1263268763279],
              [0.0, 236.36510731401265, 228.35450912275593],
              [0.0, 0.0, 1.0]])
D = np.array([-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566])
DIM = (640, 480)


# 以逆时针转动为正方向
def cal_RT(x_a=90., y_a=0., z_a=0., x_t=0., y_t=0., z_t=0.):
    # 创建单体矩阵
    Rx = np.zeros([3, 3], dtype=np.float)
    Ry = np.zeros([3, 3], dtype=np.float)
    Rz = np.zeros([3, 3], dtype=np.float)
    # 先转换为弧度制
    x_a = x_a / 180. * math.pi
    y_a = y_a / 180. * math.pi
    z_a = z_a / 180. * math.pi
    # 计算绕x旋转矩阵
    Rx[1][1] = math.cos(x_a)
    Rx[1][2] = -math.sin(x_a)
    Rx[2][1] = math.sin(x_a)
    Rx[2][2] = math.cos(x_a)
    Rx[0][0] = 1.
    # 计算绕y旋转矩阵
    Ry[0][0] = math.cos(y_a)
    Ry[0][2] = -math.sin(y_a)
    Ry[2][0] = math.sin(y_a)
    Ry[2][2] = math.cos(y_a)
    Ry[1][1] = 1.
    # 计算绕z旋转矩阵
    Rz[0][0] = math.cos(z_a)
    Rz[0][1] = -math.sin(z_a)
    Rz[1][0] = math.sin(z_a)
    Rz[1][1] = math.cos(z_a)
    Rz[2][2] = 1.
    # 计算混合R旋转矩阵
    R = np.dot(Rx, Ry)
    R = np.dot(Rz, R)
    # R = np.dot(Ry, Rx)
    # R = np.dot(Rz, R)

    # 创建总矩阵
    RT = np.zeros([4, 4], dtype=np.float)
    RT[:3, :3] = R
    RT[0][3] = x_t
    RT[1][3] = y_t
    RT[2][3] = z_t
    RT[3][3] = 1
    # print('Rotation+Transform:\n', RT)
    return RT


# 输入为一个三维坐标点，输出为一个二维int型坐标
def undistorePoint(inputPoint):
    point = np.zeros([1, 1, 3], dtype=np.float64)
    point[0][0] = inputPoint[:3]
    rvecs = np.zeros((3, 1), dtype=np.float64)
    tvecs = np.zeros((3, 1), dtype=np.float64)
    distoredPoint, jacobian = cv.fisheye.projectPoints(point, rvecs, tvecs, K, D)
    distoredPoint = (int(distoredPoint[0][0][0]), int(distoredPoint[0][0][1]))
    return distoredPoint


def nothing(x):
    pass


barPic = cv.imread('D:\\show_pic\\desktop2.png', 1)
cv.imshow('BAR', barPic)
cv.createTrackbar('Front', 'BAR', 0, 500, nothing)
cv.createTrackbar('Right', 'BAR', 0, 500, nothing)
cv.createTrackbar('Back', 'BAR', 0, 500, nothing)
cv.createTrackbar('Left', 'BAR', 0, 500, nothing)
cv.setTrackbarPos('Front', 'BAR', 320)
cv.setTrackbarPos('Right', 'BAR', 471)
cv.setTrackbarPos('Back', 'BAR', 402)
cv.setTrackbarPos('Left', 'BAR', 320)

# 这个函数是微调两个摄像头的数据，对两个摄像头联合标定
_front = cv.imread("../pictures/PSProcess/4_front.jpg", 1)
_right = cv.imread("../pictures/PSProcess/4_right.jpg", 1)
_back = cv.imread("../pictures/PSProcess/4_back.jpg", 1)
_left = cv.imread("../pictures/PSProcess/4_left.jpg", 1)
x_offset = 300
y_offset = 400
objPoint1 = [x_offset, y_offset, 0, 1]  # 设定一个在地面上的点，右前
objPoint2 = [x_offset, -y_offset, 0, 1]  # 设定一个在地面上的点，右后
objPoint3 = [-x_offset, -y_offset, 0, 1]  # 设定一个在地面上的点，左后
objPoint4 = [-x_offset, y_offset, 0, 1]  # 设定一个在地面上的点，左前
# 矩阵是先转再按照转动后的轴去平移，所以从世界->相机时，T的构建就很麻烦，所以我
# 使用从摄像机->世界的方式，之后再求逆
# 这里还有一个需要注意的问题，就是在往负半轴平移的时候，写入的是正值

angle = [32., 32., 32., 32.]
from getCalibrationPhotos import jointFourImages
while True:
    front = copy.copy(_front)
    right = copy.copy(_right)
    back = copy.copy(_back)
    left = copy.copy(_left)
    angle[0] = cv.getTrackbarPos('Front', 'BAR') / 10.
    angle[1] = cv.getTrackbarPos('Right', 'BAR') / 10.
    angle[2] = cv.getTrackbarPos('Back', 'BAR') / 10.
    angle[3] = cv.getTrackbarPos('Left', 'BAR') / 10.
    # 处理前方摄像头
    mtxFront = cal_RT(x_a=-(90.0+angle[0]), y_t=200.0, z_t=180.0)
    mtxFront = np.linalg.inv(mtxFront)
    print(mtxFront)
    # 处理右方摄像头
    mtxRight = cal_RT(x_a=-(90.0+angle[1]), z_a=-90., x_t=170.0, z_t=175.0)
    mtxRight = np.linalg.inv(mtxRight)
    # 处理后方摄像头
    mtxBack = cal_RT(x_a=-(90.0+angle[2]), z_a=-180., y_t=-200.0, z_t=175.0)
    mtxBack = np.linalg.inv(mtxBack)
    # 处理左方摄像头
    mtxLeft = cal_RT(x_a=-(90.0+angle[3]), z_a=90., y_t=40., x_t=-170.0, z_t=175.0)
    mtxLeft = np.linalg.inv(mtxLeft)

    camPoint1Front = np.dot(mtxFront, objPoint1)
    camPoint1Right = np.dot(mtxRight, objPoint1)
    print(objPoint1, camPoint1Front)

    camPoint2Right = np.dot(mtxRight, objPoint2)
    camPoint2Back = np.dot(mtxBack, objPoint2)

    camPoint3Back = np.dot(mtxBack, objPoint3)
    camPoint3Left = np.dot(mtxLeft, objPoint3)

    camPoint4Left = np.dot(mtxLeft, objPoint4)
    camPoint4Front = np.dot(mtxFront, objPoint4)

    uvPoint = undistorePoint(camPoint1Front)
    cv.circle(front, uvPoint, 5, (0, 200, 255), -1)
    uvPoint = undistorePoint(camPoint1Right)
    cv.circle(right, uvPoint, 5, (0, 200, 255), -1)

    uvPoint = undistorePoint(camPoint2Right)
    cv.circle(right, uvPoint, 5, (255, 10, 255), -1)
    uvPoint = undistorePoint(camPoint2Back)
    cv.circle(back, uvPoint, 5, (255, 10, 255), -1)

    uvPoint = undistorePoint(camPoint3Back)
    cv.circle(back, uvPoint, 5, (100, 100, 255), -1)
    uvPoint = undistorePoint(camPoint3Left)
    cv.circle(left, uvPoint, 5, (100, 100, 255), -1)

    uvPoint = undistorePoint(camPoint4Left)
    cv.circle(left, uvPoint, 5, (255, 255, 255), -1)
    uvPoint = undistorePoint(camPoint4Front)
    cv.circle(front, uvPoint, 5, (255, 255, 255), -1)
    jointFourImages(front, left, back, right)
    # cv.imshow("Front", front)
    # cv.imshow("Right", right)
    # cv.imshow("Back", back)
    # cv.imshow("Left", left)
    cv.waitKey(10)


# 这个是指定世界坐标系下的点画在图片上
img = cv.imread("../pictures/PSProcess/4_front.jpg", 1)
# for x in range(-700, 700, 50):
for x in range(0, 50):
    objPoint = [x, 60, 0, 1]  # 设定一个在地面上的点
    mtx = cal_RT(x_a=-(90.0+32), z_a=x, z_t=180.0)
    mtx = np.linalg.inv(mtx)
    camPoint = np.dot(mtx, objPoint)
    ucPoint = undistorePoint(camPoint)
    cv.circle(img, ucPoint, 5, (0, 200, 255), -1)
    print('objPoint:', objPoint)
    print('camPoint:', camPoint)
    print('mtx:', mtx)
    cv.imshow("draw", img)
    cv.waitKey(0)




