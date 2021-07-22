# 这个函数是通过透视变换拼接二维全景图
import cv2 as cv
import numpy as np
import copy
from undistortionCamera import init_K_D


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

mtx_list = [RT_front, RT_left, RT_back, RT_right]
K, D, DIM = init_K_D()
rvecs = np.zeros((3, 1), dtype=np.float64)
tvecs = np.zeros((3, 1), dtype=np.float64)


def mousecallback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        print('[', x, ',', y, '],')


def undistort(img, num):
    global DIM, K, D
    k = K[num]
    d = D[num]
    new_mtx, ROI = cv.getOptimalNewCameraMatrix(k, d, DIM, 1, DIM)
    map1, map2 = cv.fisheye.initUndistortRectifyMap(k, d, np.eye(3), new_mtx, DIM, cv.CV_16SC2)
    undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    print(map1)
    return undistorted_img, map1, map2


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


# 返回是以u、v形式返回的
def fucking_find(img, u, v, map1):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if u-10 < map1[i][j][0] < u + 10:
                if v-10 < map1[i][j][1] < v+10:
                    print('找到候选点')
                    return j, i


# 函数为字面意思，就是得到这个方向上的透视变换图像
def get_corner_perspective_image(img, cam, x_list, y_list):
    un_img, map1, _ = undistort(img, cam)
    world_point = np.zeros([4, 4], np.float)
    world_point[0] = [x_list[0], y_list[0], 0, 1]  # 左下角
    world_point[1] = [x_list[0], y_list[1], 0, 1]  # 左上角
    world_point[2] = [x_list[1], y_list[0], 0, 1]  # 右下角
    world_point[3] = [x_list[1], y_list[1], 0, 1]  # 右上角
    extric_mtx = mtx_list[0]
    uv_point = ProjectPoint(world_point, extric_mtx, 4, 0)  # 后两个参数分别是处理点的个数和相机index
    point_src = np.zeros([4, 2], dtype=np.float32)
    for i in range(4):
        u = uv_point[i][0]
        v = uv_point[i][1]
        wanted = fucking_find(img, u, v, map1)
        point_src[i] = [wanted[0], wanted[1]]  # 通过查找的方式获取已经获得的相机原图的点在去畸变图中的位置
        # 打开下面的语句就可逐过程显示
        cv.circle(img, (u, v), 3, (0, 255, 200), 2)
        cv.circle(un_img, (wanted[0], wanted[1]), 3, (0, 25, 200), 2)
        print(img.shape)
        cv.imshow('s', img)
        cv.imshow('d', un_img)
        print('map1', map1[v][u])
        cv.setMouseCallback('d', mousecallback)
        cv.setMouseCallback('s', mousecallback)
    # h, w = img.shape[:-1]
    w = int(abs(x_list[1]-x_list[0]))
    h = int(abs(y_list[1]-y_list[0]))
    point_dst = np.array([[0, h], [0, 0], [w, h], [w, 0]], dtype=np.float32)
    TRANS_MTX = cv.getPerspectiveTransform(point_src, point_dst)
    after = cv.warpPerspective(un_img, TRANS_MTX, (w, h))
    cv.imshow('plant', after)
    cv.waitKey(0)
    return after


if __name__ == '__main__':
    name = 'E:/WORKPLACE/3DSurround/pictures/joint/' + str(3) + '_right.jpg'
    img = cv.imread(name, 1)
    get_corner_perspective_image(img, cam=3, x_list=[-330, 330], y_list=[250, 550])
    un_img, map1, _ = undistort(img, 0)
    cv.imwrite('E:\\WORKPLACE\\3DSurround\\pictures\\PSProcess\\4.jpg', un_img)
    cv.imshow('hhh1', img)
    cv.imshow('hhh', un_img)
    cv.waitKey(0)