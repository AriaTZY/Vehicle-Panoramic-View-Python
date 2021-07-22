# 这个程序应该是专门用来提供uv坐标的拾取工作的，配备了callback函数
import cv2 as cv
import numpy as np
from matplotlib import pyplot
from undistortionCamera import init_K_D

emun = [['F', 0], ['L', 1], ['B', 2], ['R', 3]]
index = 1


def mousecallback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        print('[', x, ',', y, '],')
        undis_uv = caculate_undistorted([x, y], K[index], D[index])
        print(undis_uv)


# 计算前摄像头的点在相机坐标系下的表示，有一个特点就是x坐标是相同的
# 输入是在图像中的点和x值，输出是相机坐标系下xyz
def caculate_camera_xyz(uv):
    uvPoint = np.zeros([1, 1, 2], np.float)
    uvPoint[0][0] = uv
    xy = cv.fisheye.undistortPoints(uvPoint, K, D)
    # print(uv, xy)
    # 利用相等的x坐标去求真实的xyz坐标
    a = xy[0][0][0]
    b = xy[0][0][1]
    return [a, b, 1]


def caculate_undistorted(uv, K, D):
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


# 全自动进行特征点提取及打印，第二个index是用来说明需要用第几个K和D
def auto_print(mode, index, param):
    K, D, DIM = param
    K = K[index]
    D = D[index]
    img = cv.imread('E:/WORKPLACE/3DSurround/pictures/joint/40.jpg', 1)
    ret, corners = cv.findChessboardCorners(img, (6, 9))
    worldPoint = get_world_xyz(mode)
    for i in range(54):
        cv.circle(img, (int(corners[i][0][0]), int(corners[i][0][1])), 3, (255, 255, 0), 2)
        print('[', worldPoint[i][0], ',', worldPoint[i][1], ']')
        cv.imshow('test', img)
        cv.waitKey(10)
    undistortedPoint = np.zeros([54*2], np.float)
    for i in range(54):
        x, y, z = caculate_undistorted(corners[i][0], K, D)
        undistortedPoint[i * 2 + 0] = x
        undistortedPoint[i * 2 + 1] = y
    print('world xy:')
    for i in range(54):
        print(worldPoint[i][0], ',', worldPoint[i][1], end=';')
    print('\nimage uv:')
    index = 0
    for i in range(54):
        print(undistortedPoint[i*2+0], ',', undistortedPoint[i*2+1], end=', ')
        index += 2
        if i % 6 == 0 and i != 0:
            print('...')
    print(index)
    cv.imshow('chessBoard', img)
    cv.setMouseCallback('chessBoard', mousecallback)
    cv.waitKey(0)


def get_world_xyz(mode):
    side = 26.5
    worldPoint = np.zeros([54, 2], np.float)
    index = 0
    if mode == 'B':
        # back的世界坐标系
        temp_x = -4 * side
        for x in range(9):
            temp_y = -300-6*side
            for y in range(6):
                worldPoint[index] = [temp_x, temp_y]
                index += 1
                temp_y += side
            temp_x += side
    elif mode == 'R':
        # right的世界坐标系
        temp_y = 4 * side
        for y in range(9):
            temp_x = 300 + side  # 330
            for x in range(6):
                worldPoint[index] = [temp_x, temp_y]
                index += 1
                temp_x += side
            temp_y -= side
    elif mode == 'F':
        # front的世界坐标系
        temp_x = 4 * side
        for x in range(9):
            temp_y = 300+6*side
            for y in range(6):
                worldPoint[index] = [temp_x, temp_y]
                index += 1
                temp_y -= side
            temp_x -= side
    elif mode == 'L':
        # left的世界坐标系
        temp_y = 4 * side
        for y in range(9):
            temp_x = -300-6*side
            for x in range(6):
                worldPoint[index] = [temp_x-8, temp_y]
                index += 1
                temp_x += side
            temp_y -= side
    return worldPoint


if __name__ == '__main__':
    K, D, DIM = init_K_D()
    auto_print(emun[index][0], emun[index][1], [K, D, DIM])