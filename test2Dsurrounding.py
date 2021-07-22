# 这个程序文件是测试2d全景拼接效果的

import cv2 as cv
import numpy as np
from undistortionCamera import init_K_D

# ====================这里是常参数定义区==========================
# K = np.array([[[238.87052313877663, 0.0, 328.1263268763279],[0.0, 236.36510731401265, 228.35450912275593],[0.0, 0.0, 1.0]],
#               [[238.87052313877663, 0.0, 328.1263268763279], [0.0, 236.36510731401265, 228.35450912275593],[0.0, 0.0, 1.0]],
#               [[238.87052313877663, 0.0, 328.1263268763279], [0.0, 236.36510731401265, 228.35450912275593],[0.0, 0.0, 1.0]],
#               [[238.87052313877663, 0.0, 328.1263268763279], [0.0, 236.36510731401265, 228.35450912275593],[0.0, 0.0, 1.0]]])
# D = np.array([[-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566],
#               [-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566],
#               [-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566],
#               [-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566]])
# DIM = (640, 480)

K, D, DIM = init_K_D()
rvecs = np.zeros((3, 1), dtype=np.float64)
tvecs = np.zeros((3, 1), dtype=np.float64)
# ====================四个块在世界坐标系下=======================
# RT_front = np.array([[-0.9867,    0.0401,    0.0507,    0.6864],
#    [-0.0178,    0.4885,   -0.8751, -271.1025],
#    [-0.0716,   -0.8839,   -0.4813,  113.5791]])
#
#
# RT_right = np.array([[0.0000,   0.9792,    0.0482,    5.7428],
#     [0.6876,    0.0074,   -0.7371, -260.4682],
#     [-0.7528,    0.0620,   -0.6733,  17.6163]])
#
#
# RT_back = np.array([[0.9547, - 0.0106, - 0.0133, - 6.4997],
#                     [-0.0226, - 0.5610, - 0.8467, - 271.0703],
#                     [0.0121, 0.8868, - 0.5359, 114.6582]])
#
#
# RT_left = np.array([[-0.0514,   -0.9868 ,   0.0613,   -2.9813],
#    [-0.6216 ,  -0.0199 ,  -0.7883 ,-247.4919],
#     [0.7951 ,  -0.0731 ,  -0.6124 ,  30.4715]])  # 1


# 使用棋盘格标定的
# RT_back = np.array([[0.9794,   -0.0285,   -0.0140,  -13.3692],
#    [-0.0265,   -0.5557 ,  -0.8388, -270.1422],
#     [0.0158,    0.8560,   -0.5450,  106.9254]])
#
# RT_left = np.array([[-0.0546,   -0.9829,    0.0581,   -3.6427],
#    [-0.6170,   -0.0118,   -0.7937, -244.2275],
#     [0.8032,   -0.0788,   -0.6058,   36.2440]])
#
# RT_front = np.array([[-0.9783,    0.0415,    0.0462,    0.1098],
#    [-0.0177,    0.4969,   -0.8735, -270.7543],
#    [-0.0613,   -0.8903,   -0.4854,  119.3557]])
#
#
# RT_right = np.array([[0.0271,    0.9821,    0.0347,   -3.7300],
#     [0.6697,    0.0084,   -0.7531, -254.0187],
#    [-0.7657,    0.0422,   -0.6574,   33.1556]])


# 最新一批的相机外参，使用了不同相机不同内参以及畸变参数来做的
RT_front = np.array([[-0.994994898660732, 	0.0159468221122663,	0.0333726532105878,	-0.664385300205282],
[-0.0186089549879460,	0.560210248843327,	-0.829790788127661,	-283.571411187421],
[-0.0318861507994662,	-0.833453826214823,	-0.557109586081681,	112.492147199740]])

RT_left = np.array([[0.00481150623611693,	-1.00224836515249,	0.0232894154011142,	1.10682209804278],
[-0.674784714696795,	-0.0197516465139249,	-0.736176897177524,	-257.097460589928],
[0.734587881252186,	-0.0130116984661592,	-0.676396912305129,	36.3343098961476]])

RT_back = np.array([[1.01411933629307,	-0.0189188882285207,	-0.00214321031915822,	-2.63373806551220],
[-0.0116268377102352,	-0.565395923023170,	-0.819371982618422,	-274.933977726956],
[0.0128194701474831,	0.807724912819171,	-0.573598905042153,	112.376058308863]])


RT_right = np.array([[0.000588383270798021,	1.01130105691855,	0.0148622290291620,	0.342314980399614],
[0.715961431631093,	0.0123756783440439,	-0.689769010317032,	-261.140845235426],
[-0.682055782001940,	0.00896881560195422,	-0.724045270879343,	25.8994849077818]])



# 提供点的映射，输入是世界坐标，输出是图像的坐标
# 输入的世界坐标点一定是
def ProjectPoint(cornerArray, mtx, col_num, KD_index):
    k = K[KD_index]
    d = D[KD_index]
    cameraPointArray = np.dot(mtx, cornerArray.T)
    cameraPointArray = np.reshape(cameraPointArray.T, [col_num, 1, 3])
    ImageUV, jacobian = cv.fisheye.projectPoints(cameraPointArray, rvecs, tvecs, k, d)
    ImageUV = np.reshape(ImageUV, [col_num, 2])
    ImageUV = np.array(ImageUV, np.int)
    return ImageUV


import time
def renderZone(sheet, img, mtx, boarder, map, scale, KD_index):
    # 首先把图片拓扑到一个更大的图片上，防止访问超限
    biggerImg = np.zeros([img.shape[0] * 2, img.shape[1] * 2, 3], np.uint8)
    biggerImg[:img.shape[0], :img.shape[1]] = img
    width = (boarder[0][1] - boarder[0][0]) // scale
    height = (boarder[1][1] - boarder[1][0]) // scale
    cornerArray = np.zeros([width, 4], np.float)
    caculation_time = 0.
    render_time = 0.
    for y_index in range(height):
        check1 = time.time()
        for x_index in range(width):
            row = boarder[1][0] + y_index * scale  # 计算出实际世界坐标系下的xy值
            col = boarder[0][0] + x_index * scale
            cornerArray[x_index][0] = col
            cornerArray[x_index][1] = row
            cornerArray[x_index][2] = 0
            cornerArray[x_index][3] = 1
        LUT = ProjectPoint(cornerArray, mtx, width, KD_index)
        check2 = time.time()
        caculation_time += check2 - check1
        render_row = (map[1][1] - boarder[1][1])//scale - y_index + height - 1
        for draw_u in range(width):
            render_col = (map[0][1] + boarder[0][0] + draw_u * scale)//scale
            sheet[render_row][render_col] = biggerImg[LUT[draw_u][1]][LUT[draw_u][0]]
        check3 = time.time()
        render_time += check3 - check2
    # print('caculation time:', caculation_time)
    # print('rendering time:', render_time)
    return sheet


# 渲染重合部分
def renderOverLappingZone(sheet, img1, img2):
    import math
    from matplotlib import pyplot
    img1 = cv.resize(img1, (430, 450))
    img2 = cv.resize(img2, (430, 450))
    # 创建权重矩阵
    weight = np.zeros([450, 430], np.float)
    zone = np.zeros([450, 430, 3], np.uint8)
    half_pi = math.pi / 2
    for u in range(430):
        for v in range(450):
            x = 429 - u
            y = 449 - v
            if y == 0:
                y = 0.1
            theta = math.atan(x / y)
            # if theta < half_pi/2:
            #     weightt = 0
            # else:
            #     weightt = 1
            weightt = (theta / half_pi) * 1.
            weight[v][u] = (theta / half_pi) * 1.
            zone[v][u] = np.array(img1[v][u]*weightt + img2[v][u]*(1-weightt), np.uint8)
    cv.imshow('1', img1)
    cv.imshow('2', img2)
    zone = cv.resize(zone, (0, 0), None, 0.5, 0.5)
    cv.imshow('zone', zone)
    cv.waitKey(0)
    pyplot.imshow(weight, 'gray')
    pyplot.show()


if __name__ == '__main__':
    import copy
    scale = 2  # 缩放比例

    # 这个是两鬓覆盖上下图像的模式
    boarder_ALL = [[-600, 600], [-700, 700]]  # 整体的显示范围，显示x范围，后是y范围
    boarder_F = [[-600, 600], [250, 700]]  # front的显示范围
    boarder_R = [[170, 600], [-700, 700]]  # right的显示范围
    boarder_B = [[-600, 600], [-700, -250]]  # back的显示范围
    boarder_L = [[-600, -170], [-700, 700]]

    # 这个是上下覆盖两鬓图像的模式
    # boarder_ALL = [[-600, 600], [-700, 700]]  # 整体的显示范围，显示x范围，后是y范围
    # boarder_F = [[-600, 600], [250, 700]]  # front的显示范围
    # boarder_R = [[170, 600], [-500, 500]]  # right的显示范围
    # boarder_B = [[-600, 600], [-700, -250]]  # back的显示范围
    # boarder_L = [[-600, -170], [-500, 500]]


    # boarder_ALL = [[-900, 900], [-1000, 1000]]  # 整体的显示范围，显示x范围，后是y范围
    # boarder_F = [[-900, 900], [250, 1000]]  # front的显示范围
    # boarder_R = [[170, 900], [-1000, 1000]]  # right的显示范围
    # boarder_B = [[-900, 900], [-1000, -250]]  # back的显示范围
    # boarder_L = [[-900, -170], [-1000, 1000]]

    # front = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/5_front.jpg", 1)
    # right = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/5_right.jpg", 1)
    # back = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/5_back.jpg", 1)
    # left = cv.imread("E:/WORKPLACE/3DSurround/pictures/joint/5_left.jpg", 1)

    img_num = '0'
    front = cv.imread("E:/WORKPLACE/3DSurround/pycharm/image/front/img_"+img_num+".jpg", 1)
    right = cv.imread("E:/WORKPLACE/3DSurround/pycharm/image/right/img_"+img_num+".jpg", 1)
    back = cv.imread("E:/WORKPLACE/3DSurround/pycharm/image/back/img_"+img_num+".jpg", 1)
    left = cv.imread("E:/WORKPLACE/3DSurround/pycharm/image/left/img_"+img_num+".jpg", 1)
    # 按照缩放要求和显示范围新建画布
    sheet_width = (boarder_ALL[0][1]-boarder_ALL[0][0])//scale
    sheet_height = (boarder_ALL[1][1]-boarder_ALL[1][0])//scale
    sheet = np.zeros([sheet_height, sheet_width, 3], np.uint8)

    # 开始渲染
    sheet = renderZone(sheet, front, RT_front, boarder_F, boarder_ALL, scale, 0)
    zoneFromFront = copy.copy(sheet[0:450//scale, 0:430//scale])
    sheet = renderZone(sheet, back, RT_back, boarder_B, boarder_ALL, scale, 2)

    sheet = renderZone(sheet, right, RT_right, boarder_R, boarder_ALL, scale, 3)
    sheet = renderZone(sheet, left, RT_left, boarder_L, boarder_ALL, scale, 1)
    zoneFromLeft = copy.copy(sheet[0:450//scale, 0:430//scale])
    # zoneFromLeft = copy.copy(sheet[(1400-450)//scale:1400//scale, 0:430//scale])

    # cv.imshow('front', zoneFromFront)
    # cv.imshow('right', zoneFromLeft)
    # cv.imwrite('E:/WORKPLACE/3DSurround/pictures/geometric/pre_zone_1_1.jpg', zoneFromLeft)
    # cv.imwrite('E:/WORKPLACE/3DSurround/pictures/geometric/pre_zone_1_2.jpg', zoneFromFront)
    # # zoneFromLeft = sheet[0:450//2, 0:430//2]
    # renderOverLappingZone(sheet, zoneFromLeft, zoneFromFront)
    print(sheet.shape)
    cv.imshow('final', sheet)

    cv.waitKey(0)

