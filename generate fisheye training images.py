# 用计算机计算生成结合成多张鱼眼照片进行训练。

import math
import cv2 as cv
import numpy as np
from undistortionCamera import init_K_D


K, D, DIM = init_K_D()
rvecs = np.zeros((3, 1), dtype=np.float64)
tvecs = np.zeros((3, 1), dtype=np.float64)


def tesd(img, ratio=0.5, center_ratio=0.5):
    rows, cols, _ = img.shape
    fish_image = np.zeros([rows, cols, 3], dtype=np.uint8)
    # 调整这个中心点的值可以帮助改变图像畸变后的映射位置
    u0 = cols*center_ratio
    v0 = rows/2
    # 调整这个ratio有助于生成更加鱼眼畸变的相机，ratio越小畸变越明显
    sigma_u = u0*ratio
    sigma_v = v0*ratio
    k = D[1]  # 随便取一个相机的畸变参数即可

    def project_p(col, row):
        x = (col - u0) / sigma_u
        y = (row - v0) / sigma_v
        r = math.sqrt(x ** 2 + y ** 2)
        theta = math.atan(r)
        theta_d = theta * (
                    1 + k[0] * pow(theta, 2) + k[1] * pow(theta, 4) + k[2] * pow(theta, 6) + k[3] * pow(theta, 8))
        if theta_d != 0:
            xd = theta_d * x / r
            yd = theta_d * y / r
        else:
            xd = 0
            yd = 0
        ud = int(xd * sigma_u + u0)
        vd = int(yd * sigma_v + v0)
        return ud, vd

    # 计算每个点的映射
    for col in range(cols):
        for row in range(rows):
            ud, vd = project_p(col, row)
            fish_image[vd][ud] = img[row][col]

    # 计算原图中四个角映射后的坐标，用于裁剪
    edge = np.zeros([4, 2], dtype=int)
    edge[0] = project_p(0, 0)  # 左上角
    edge[1] = project_p(cols, 0)  # 右上角
    edge[2] = project_p(0, rows)  # 左下角
    edge[3] = project_p(cols, rows)  # 右下角角

    for i in range(4):
        cv.circle(fish_image, (edge[i][0], edge[i][1]), 5, color=(0, 255, 250), thickness=-1)
    out = fish_image[edge[0][1]:edge[3][1], edge[0][0]:edge[3][0]]
    return out, fish_image


if __name__ == '__main__':
    # for i in range(5):
    #     name = 'image/training data generator/'+ str(i) + '.jpg'
    #     img = cv.imread(name, 1)
    #     img = cv.resize(img, (0, 0), None, 0.3, 0.3)
    #     cv.imshow('TESE', img)
    #     cv.imwrite(name, img)

    img = cv.imread('image/training data generator/4.jpg', 1)
    img = cv.resize(img, (0, 0), None, 0.5, 0.5)
    # 进行patch，因为照片里车照的有点大了
    ratio = 1.3
    rows, cols, _ = img.shape
    patched_img = np.ones([int(rows*ratio), int(cols*ratio), 3], dtype=np.uint8)
    patched_img = patched_img * 255
    start_u = (patched_img.shape[0] - rows)//2
    start_v = (patched_img.shape[1] - cols)//2
    patched_img[start_v:start_v+rows, start_u:start_u+cols, :] = img

    out, fish_img = tesd(patched_img, center_ratio=0.5)
    cv.imshow('TESE', fish_img)
    cv.imshow('TESE2', out)

    # save_path = 'image/training data generator/logout/diff_central_7.jpg'
    # cv.imwrite(save_path, fish_img)
    cv.waitKey(0)