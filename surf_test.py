import cv2 as cv
import numpy as np
import copy
from undistortionCamera import init_K_D
from matplotlib import pyplot
import math

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
def fucking_find(u, v, map1):
    for i in range(front.shape[0]):
        for j in range(front.shape[1]):
            if u-10 < map1[i][j][0] < u + 10:
                if v-10 < map1[i][j][1] < v+10:
                    print('找到候选点')
                    return j, i


# 聚类
def cluster(data, t_range):
    num = len(data)
    max_cluster_data_num = 0
    best_avr = 0
    for i in range(num):  # 遍历每一个点
        cluster_data_num = 0  # 这个点为中心时，有多少个点被包括
        average_data = 0.0  # 平均值
        for j in range(num):  # 求每一个点的包含范围
            if math.fabs(data[i]-data[j]) < t_range:
                cluster_data_num += 1
                average_data += data[j]
        average_data = average_data/cluster_data_num
        print('第',i,'点，据类点个数：',cluster_data_num, '平均值', average_data)
        if cluster_data_num > max_cluster_data_num:
            max_cluster_data_num = cluster_data_num
            best_avr = average_data
        print('  当前最优平均值：', best_avr)
    return best_avr


# 函数为字面意思，就是得到这个方向上的透视变换图像
def get_corner_perspective_image(img, cam, pt1, pt2):
    un_img, map1, _ = undistort(img, cam)
    world_point = np.zeros([4, 4], np.float)
    zlist = [0, -300]
    world_point[0] = [pt1[0], pt1[1], zlist[0], 1]  # 左下角
    world_point[1] = [pt1[0], pt1[1], zlist[1], 1]  # 左上角
    world_point[2] = [pt2[0], pt2[1], zlist[0], 1]  # 右下角
    world_point[3] = [pt2[0], pt2[1], zlist[1], 1]  # 右上角
    extric_mtx = mtx_list[cam]
    uv_point = ProjectPoint(world_point, extric_mtx, 4, 0)
    point_src = np.zeros([4, 2], dtype=np.float32)
    for i in range(4):
        u = uv_point[i][0]
        v = uv_point[i][1]
        wanted = fucking_find(u, v, map1)
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
        cv.waitKey(0)
    # h, w = img.shape[:-1]
    w = int(math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2))
    h = int(math.fabs(zlist[1]-zlist[0]))
    point_dst = np.array([[0, h], [0, 0], [w, h], [w, 0]], dtype=np.float32)
    TRANS_MTX = cv.getPerspectiveTransform(point_src, point_dst)
    after = cv.warpPerspective(un_img, TRANS_MTX, (w, h))
    return after


# 使用创建LUT的方式做
def get_corner_perspective_image2(img, cam, pt_center, fun_k, wide):
    h = 300  # 截取高度
    w = wide*2
    fun_b = pt_center[1]-fun_k*pt_center[0]
    map_x = np.zeros([h, w], np.float32)
    map_y = np.zeros([h, w], np.float32)
    start_x = pt_center[0]-wide
    for i in range(h):
        for j in range(w):
            world_point = np.array([start_x + j, fun_b + fun_k*(start_x + j), -i, 1])
            uv = ProjectPoint(world_point, mtx_list[cam], 1, cam)
            map_x[h - 1 - i][j] = uv[0][0]
            map_y[h - 1 - i][j] = uv[0][1]
    a = cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    save_map = [map_x, map_y]
    np.save('map_data/0_right_map.npy', save_map)
    return a


if __name__ == '__main__':
    # [Rmap_x, Rmap_y] = np.load('map_data/0_right_map.npy')
    # [Fmap_x, Fmap_y] = np.load('map_data/0_front_map.npy')
    # for j in range(3):
    #     i = j+13
    #     name = 'E:/WORKPLACE/3DSurround/pictures/joint/'+str(i)+'_right.jpg'
    #     right = cv.imread(name, 1)
    #     name = 'E:/WORKPLACE/3DSurround/pictures/joint/' + str(i) + '_front.jpg'
    #     front = cv.imread(name, 1)
    #     # a = get_corner_perspective_image2(right, 3, (600, 700), -1, 400)
    #     right_roi = cv.remap(right, Rmap_x, Rmap_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    #     front_roi = cv.remap(front, Fmap_x, Fmap_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    #     cv.imshow('p', right_roi)
    #     cv.imshow('p2', front_roi)
    #     name = 'E:/WORKPLACE/3DSurround/pictures/surf/'+str(i*2+0)+'.jpg'
    #     cv.imwrite(name, front_roi)
    #     name = 'E:/WORKPLACE/3DSurround/pictures/surf/'+str(i*2+1)+'.jpg'
    #     cv.imwrite(name, right_roi)
    #     print('保存成功')
    #     cv.waitKey(0)

    right = cv.imread('E:/WORKPLACE/3DSurround/pictures/surf/26.jpg', 1)
    front = cv.imread('E:/WORKPLACE/3DSurround/pictures/surf/27.jpg', 1)
    right = cv.resize(right, (0, 0), None, 0.5, 0.5)
    front = cv.resize(front, (0, 0), None, 0.5, 0.5)
    # right = right[:, front.shape[1]//2:front.shape[1], :]
    # front = front[:, front.shape[1]//2:front.shape[1], :]
    right = right[right.shape[0]*0//3:right.shape[0]*3//3, :, :]
    front = front[front.shape[0]*0//3:front.shape[0]*3//3, :, :]
    surf = cv.xfeatures2d.SURF_create()
    # 进行各自的特征点检测
    kp1, des1 = surf.detectAndCompute(front, None)
    kp2, des2 = surf.detectAndCompute(right, None)
    # -----------------------进行左边部分的标定-------------------------- #
    # 匹配，具体匹配的参数描述详见 https://blog.csdn.net/weixin_44072651/article/details/89262277
    bf = cv.BFMatcher()
    good = []
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(front, kp1, right, kp2, matches[:20], None, flags=2)  # 画出匹配关系
    cv.imshow('s', img3)
    cv.waitKey(0)
    print(matches[0].queryIdx)
    print(matches[0].trainIdx)
    differ = []
    differ2 = []
    num = min(20, len(matches))
    for i in range(num):
        index1 = matches[i].queryIdx
        index2 = matches[i].trainIdx
        differ.append(kp1[index1].pt[0]-kp2[index2].pt[0])
        differ2.append(kp1[index1].pt[1] - kp2[index2].pt[1])
        cv.circle(front, (int(kp1[index1].pt[0]), int(kp1[index1].pt[1])), 3, (255, 255, 255), 3)
        cv.circle(right, (int(kp2[index2].pt[0]), int(kp2[index2].pt[1])), 3, (255, 255, 0), 3)
        cv.imshow('front', front)
        cv.imshow('right', right)
        cv.setMouseCallback('front', mousecallback)
        cv.waitKey(10)
    cv.waitKey(0)
    avr = cluster(differ, 50)
    pyplot.scatter(differ2, differ)
    pyplot.plot([min(differ2), max(differ2)], [avr, avr], 'r--')
    pyplot.show()


    init_thre = 0.4  # 初始化阈值
    while len(good) < 20:
        for m, n in matches:
            if m.distance < init_thre * n.distance:
                good.append([m])
        init_thre = init_thre + 0.05
    check_num = 20  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):  # 这里使用了遍历操作，index表示索引，d才是good中的match点
        temp.append(d[0].distance)  # 排列刚刚good中的欧氏距离，做进一步筛选
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        best.append(good[index])  # 这里把good进一步到best
    # 进行绘制（为UI做样子，所以多停留几秒）
    merge = np.zeros([front.shape[0], front.shape[1]*2, 3], np.uint8)
    # 开始基于匹配点做放射变换，得到一个放射变化map
    src_point = []
    dst_point = []
    x_differ = []
    y_differ = []
    for i in range(check_num):
        src_index = best[i][0].queryIdx  # 用这个来获取原图中和目标图中的关键点索引
        dst_index = best[i][0].trainIdx
        src_point.append(kp1[src_index].pt)  # 这一步是把两个对应点的坐标取出来
        dst_point.append(kp2[dst_index].pt)
        print(kp2[dst_index].pt)
        cv.circle(front, (int(kp1[src_index].pt[0]), int(kp1[src_index].pt[1])), 3, (255, 255, 0), 1)
        cv.circle(right, (int(kp2[dst_index].pt[0]), int(kp2[dst_index].pt[1])), 3, (255, 0, 255), 1)
    cv.imshow('pic', right)
    cv.imshow('pic1', front)
    cv.waitKey(0)
    for z in np.arange(0, -400, -100):
        world_point = np.array([600, 250, z, 1], np.float)
        rev = ProjectPoint(world_point, RT_right, 1, 3)
        cv.circle(right, (rev[0][0], rev[0][1]), 3, (0, 255, 100), 2)
    cv.imshow('pic', right)
    cv.waitKey(0)



# left = copy.copy(left_input)
# mid = copy.copy(mid_input)


