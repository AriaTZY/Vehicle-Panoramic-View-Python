import cv2 as cv
import numpy as np
import math


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


A = np.array([[645.0, 0.0, 405.1], [0.0, 643.79, 314.13], [0, 0, 1]], np.float)
path = 'E:/WORKPLACE/3DSurround/pictures/persp_test/test.jpg'

img = cv.imread(path, 1)
world_point = np.array([[300, 1700, 0, 1], [-300, 1700, 0, 1], [300, 1000, 0, 1], [-300, 1000, 0, 1]])
mtx = cal_RT(x_a=-(90.0+22), z_a=0, z_t=1000.0)
mtx_eye = cal_RT(x_a=-(90.0+22), z_a=0, z_t=1000.0, x_t=-400)
mtx = np.linalg.inv(mtx)
mtx_eye = np.linalg.inv(mtx_eye)
img_uv = np.zeros([4, 2], np.float32)
eye_uv = np.zeros([4, 2], np.float32)
for i in range(4):
    camPoint = np.dot(mtx, world_point[i])
    uv_point = np.dot(A, camPoint[:-1]/camPoint[-2])
    print(uv_point)
    img_uv[i] = [uv_point[0], uv_point[1]]
    cv.circle(img, (int(uv_point[0]), int(uv_point[1])), 3, (0, 255, 250), 2)

    eyePoint = np.dot(mtx_eye, world_point[i])
    uv_point = np.dot(A, eyePoint[:-1] / eyePoint[-2])
    print(uv_point)
    eye_uv[i] = [uv_point[0], uv_point[1]]
    cv.circle(img, (int(uv_point[0]), int(uv_point[1])), 3, (0, 255, 0), 2)
M = cv.getPerspectiveTransform(img_uv, eye_uv)
dst = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))
# cv.circle(img, (int(uv_point[0]), int(uv_point[1])), 3, (0, 255, 250), 2)
cv.imshow('show', img)
cv.imshow('show2', dst)
cv.waitKey(0)
