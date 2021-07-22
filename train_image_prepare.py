import os
import copy
import cv2 as cv
import numpy as np
from test2Dsurrounding import renderZone, K, D, DIM, RT_front, RT_right, RT_back, RT_left
from perspective_2D import mousecallback
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


RT = [RT_front, RT_left,  RT_back, RT_right]

root = 'E:/WORKPLACE/3DSurround/pycharm/image/picture_mode/'
# root = 'E:/WORKPLACE/3DSurround/pycharm/image/'
root_path = root + 'txt/'


def show_2d_map(idx):
    scale = 4  # 缩放比例

    # 这个是两鬓覆盖上下图像的模式
    boarder_ALL = [[-600, 600], [-1000, 1000]]  # 整体的显示范围，显示x范围，后是y范围
    boarder_F = [[-600, 600], [250, 1000]]  # front的显示范围
    boarder_R = [[170, 600], [-1000, 1000]]  # right的显示范围
    boarder_B = [[-600, 600], [-1000, -250]]  # back的显示范围
    boarder_L = [[-600, -170], [-1000, 1000]]

    img_num = str(idx)
    front = cv.imread(root + "front/img_" + img_num + ".jpg", 1)
    right = cv.imread(root + "right/img_" + img_num + ".jpg", 1)
    back = cv.imread(root + "back/img_" + img_num + ".jpg", 1)
    left = cv.imread(root + "left/img_" + img_num + ".jpg", 1)
    # 按照缩放要求和显示范围新建画布
    sheet_width = (boarder_ALL[0][1] - boarder_ALL[0][0]) // scale
    sheet_height = (boarder_ALL[1][1] - boarder_ALL[1][0]) // scale
    sheet = np.zeros([sheet_height, sheet_width, 3], np.uint8)

    # 开始渲染
    sheet = renderZone(sheet, front, RT_front, boarder_F, boarder_ALL, scale, 0)
    sheet = renderZone(sheet, back, RT_back, boarder_B, boarder_ALL, scale, 2)

    sheet = renderZone(sheet, right, RT_right, boarder_R, boarder_ALL, scale, 3)
    sheet = renderZone(sheet, left, RT_left, boarder_L, boarder_ALL, scale, 1)

    return sheet


# 把图像坐标系下的点映射到世界坐标系中，但是这里只能默认映射到世界坐标系的地面上
# 图像坐标系给鱼眼相机下的就可以，输出的世界坐标系是直接以mm为单位的
def image_to_world_point(u, v, cam_index):
    # 创建一个1*1*2的矩阵表示当前坐标，如果多点第一维度用N，但第二维度的1总是没用的
    distorted = np.array([[[u, v]]], np.float)
    # 使用opencv自带的映射点函数把畸变的点映射回小孔模型中，这里默认是在焦距为1下的xy坐标
    get = cv.fisheye.undistortPoints(distorted, K[cam_index], D[cam_index])
    # 这里的xy还不是相机坐标系下的xy，但已经是一个物理长度了，想要真正获取相机坐标系下的点，还需要知道深度信息 Zc
    x, y = get[0][0][0], get[0][0][1]
    # 外参矩阵补全成齐次坐标，为了之后求逆
    temp = np.array([[0, 0, 0, 1]])
    homo_front = np.concatenate([RT[cam_index], temp], axis=0)
    # 求外参（齐次形式）的逆
    A = np.linalg.inv(homo_front)
    # 这个公式是一个推导，先利用zw=0的这个条件求出zc（相机坐标系下的深度信息）
    zc = -A[2][3] / (A[2][0] * x + A[2][1] * y + A[2][2])
    camera_coord = np.array([x * zc, y * zc, zc, 1])
    # 和刚刚那个齐次矩阵的逆乘起来得到
    world_coord = np.dot(A, camera_coord)
    # 返回格式 [x, y, z]
    return world_coord


# 得到当前人和汽车的坐标（世界坐标系下的坐标）
def get_bbox(idx, conf_threshold=0.9):
    vehicle = []
    person = []
    """ 进行车辆的检测部分 """
    class_name = 'vehicle'
    back_path = os.path.join(root_path, 'back', class_name, 'img_'+str(idx)+'.txt')
    front_path = os.path.join(root_path, 'front', class_name, 'img_'+str(idx)+'.txt')
    left_path = os.path.join(root_path, 'left', class_name, 'img_'+str(idx)+'.txt')
    right_path = os.path.join(root_path, 'right', class_name, 'img_'+str(idx)+'.txt')

    with open(back_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            vehicle.append([2, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    with open(front_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            vehicle.append([0, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    with open(left_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            vehicle.append([1, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    with open(right_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            vehicle.append([3, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    """ 进行行人的检测部分 """
    class_name = 'person'
    back_path = os.path.join(root_path, 'back', class_name, 'img_' + str(idx) + '.txt')
    front_path = os.path.join(root_path, 'front', class_name, 'img_' + str(idx) + '.txt')
    left_path = os.path.join(root_path, 'left', class_name, 'img_' + str(idx) + '.txt')
    right_path = os.path.join(root_path, 'right', class_name, 'img_' + str(idx) + '.txt')

    with open(back_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            person.append([2, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    with open(front_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            person.append([0, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    with open(left_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            person.append([1, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    with open(right_path) as f:
        line = f.readline().split(' ')
        if int(line[0]) and float(line[5]) > conf_threshold:  # 如果有才进行下一步
            person.append([3, float(line[1]), float(line[2]), float(line[3]), float(line[4])])

    return vehicle, person


# 得到当前人和汽车的坐标，因为用于论文的是手动标注的，所以有所不同
def get_bbox_for_road_pictuire_back_up(idx):
    vehicle = []
    person = []
    """ 进行车辆的检测部分 """
    class_name = 'vehicle'
    back_path = os.path.join(root_path, 'back', class_name, 'img_'+str(idx)+'.txt')
    front_path = os.path.join(root_path, 'front', class_name, 'img_'+str(idx)+'.txt')
    left_path = os.path.join(root_path, 'left', class_name, 'img_'+str(idx)+'.txt')
    right_path = os.path.join(root_path, 'right', class_name, 'img_'+str(idx)+'.txt')

    with open(back_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            vehicle.append([2, float(line[1]), float(line[2]), w, h])

    with open(front_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            vehicle.append([0, float(line[1]), float(line[2]), w, h])

    with open(left_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            vehicle.append([1, float(line[1]), float(line[2]), w, h])

    with open(right_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            print(float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            vehicle.append([3, float(line[1]), float(line[2]), w, h])

    """ 进行行人的检测部分 """
    class_name = 'person'
    back_path = os.path.join(root_path, 'back', class_name, 'img_' + str(idx) + '.txt')
    front_path = os.path.join(root_path, 'front', class_name, 'img_' + str(idx) + '.txt')
    left_path = os.path.join(root_path, 'left', class_name, 'img_' + str(idx) + '.txt')
    right_path = os.path.join(root_path, 'right', class_name, 'img_' + str(idx) + '.txt')

    with open(back_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            person.append([2, float(line[1]), float(line[2]), w, h])

    with open(front_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            person.append([0, float(line[1]), float(line[2]), w, h])

    with open(left_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            person.append([1, float(line[1]), float(line[2]), w, h])

    with open(right_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3]) - float(line[1])
            h = float(line[4]) - float(line[2])
            person.append([3, float(line[1]), float(line[2]), w, h])

    return vehicle, person


# 这个和backup不同的地方在于，这里默认的数据结构是index, x, y, w, h，而backup里面默认的是[index, x1, y1, x2, y2]的格式
# 所以处理方面有所不同，但是函数的作用都是一样的
def get_bbox_for_road_pictuire(idx):
    vehicle = []
    person = []
    """ 进行车辆的检测部分 """
    class_name = 'vehicle'
    back_path = os.path.join(root_path, 'back', class_name, 'img_'+str(idx)+'.txt')
    front_path = os.path.join(root_path, 'front', class_name, 'img_'+str(idx)+'.txt')
    left_path = os.path.join(root_path, 'left', class_name, 'img_'+str(idx)+'.txt')
    right_path = os.path.join(root_path, 'right', class_name, 'img_'+str(idx)+'.txt')

    with open(back_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            vehicle.append([2, float(line[1]), float(line[2]), w, h])

    with open(front_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            vehicle.append([0, float(line[1]), float(line[2]), w, h])

    with open(left_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            vehicle.append([1, float(line[1]), float(line[2]), w, h])

    with open(right_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            print(float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            w = float(line[3])
            h = float(line[4])
            vehicle.append([3, float(line[1]), float(line[2]), w, h])

    """ 进行行人的检测部分 """
    class_name = 'person'
    back_path = os.path.join(root_path, 'back', class_name, 'img_' + str(idx) + '.txt')
    front_path = os.path.join(root_path, 'front', class_name, 'img_' + str(idx) + '.txt')
    left_path = os.path.join(root_path, 'left', class_name, 'img_' + str(idx) + '.txt')
    right_path = os.path.join(root_path, 'right', class_name, 'img_' + str(idx) + '.txt')

    with open(back_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            person.append([2, float(line[1]), float(line[2]), w, h])

    with open(front_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            person.append([0, float(line[1]), float(line[2]), w, h])

    with open(left_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            person.append([1, float(line[1]), float(line[2]), w, h])

    with open(right_path) as f:
        line = f.readline().split(' ')
        if float(line[1]) != 0 and float(line[2]) != 0:  # 如果有才进行下一步
            w = float(line[3])
            h = float(line[4])
            person.append([3, float(line[1]), float(line[2]), w, h])

    return vehicle, person


if __name__ == '__main__':
    # front = cv.imread("E:/WORKPLACE/3DSurround/pycharm/image/back/img_" + str(0) + ".jpg", 1)
    # cv.imshow('front', front)
    # cv.setMouseCallback('front', mousecallback)
    # cv.waitKey(0)
    show_bbox = True
    distance_threshold = 250
    index = 0

    sum_car = '{'
    sum_person = '{'

    # 以数值形式保存的帧位置数据，只用于内部处理，不用于外部输出
    sum_car_value = []
    sum_person_value = []
    while True:
        # 从txt文件得到检测的bbox结果
        # vehicles, persons = get_bbox_for_road_pictuire(index)
        vehicles, persons = get_bbox_for_road_pictuire_back_up(index)

        # 这句可以注释，因为生成2d全景图像实在有点耗时
        # sheet = show_2d_map(index)
        # sheet = cv.cvtColor(sheet, cv.COLOR_BGR2RGB)

        # 建立3D-plot坐标
        ax = plt.subplot(121, projection='3d')
        # ax = Axes3D(fig)
        ax.set_xlim(-600, 600)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(0, 1010)

        # 建造车辆的坐标，并且绘制在图中
        car_x = np.array([-150, 150, 150, -150])
        car_y = np.array([-200, 200])
        car_x, car_y = np.meshgrid(car_x, car_y)
        car_z = np.array([[0, 0, 50, 50], [0, 0, 50, 50]])
        ax.plot_surface(car_x, car_y, car_z, color='g')

        # 打印信息
        # print('vehicle', vehicles)
        # print('person', persons)

        # 这个表示需不需要显示原始bbox的标注情况
        if show_bbox:
            img_num = str(index)
            front = cv.imread(root + "front/img_" + img_num + ".jpg", 1)
            right = cv.imread(root + "right/img_" + img_num + ".jpg", 1)
            back = cv.imread(root + "back/img_" + img_num + ".jpg", 1)
            left = cv.imread(root + "left/img_" + img_num + ".jpg", 1)
            sum_img = [front, left, back, right]

        start = time.time()
        # 保存三维坐标，但因为实际都是在地面上的估计，所以只保存平面（地面）二维坐标
        world_vehicle = []
        world_person = []
        # 这里是利用bbox计算是否有车或是行人，并投影到世界坐标上
        for vehicle in vehicles:
            cam = vehicle[0]
            print('vehicle:', vehicle)
            x = vehicle[1] + vehicle[3]/2
            y = vehicle[2] + vehicle[4]
            world_coord = image_to_world_point(x, y, cam)
            ax.scatter(world_coord[0], world_coord[1], world_coord[2], c='r', marker='o')
            world_vehicle.append([world_coord[0], world_coord[1]])
            if show_bbox:
                cv.rectangle(sum_img[cam], (int(vehicle[1]), int(vehicle[2])),
                             (int(vehicle[1]+vehicle[3]), int(vehicle[2]+vehicle[4])), (25, 25, 255), 5)

        for person in persons:
            cam = person[0]
            x = person[1] + person[3]/2
            y = person[2] + person[4]
            world_coord = image_to_world_point(x, y, cam)
            ax.scatter(world_coord[0], world_coord[1], world_coord[2], c='k', marker='o')
            world_person.append([world_coord[0], world_coord[1]])
            if show_bbox:
                cv.rectangle(sum_img[cam], (int(person[1]), int(person[2])),
                             (int(person[1]+person[3]), int(person[2]+person[4])), (25, 25, 30), 5)

        # 做类似于nms的操作，合并距离过近点，为了简化程序，只取前两点，即便检测出了很多个点也只取两点
        if len(world_vehicle) >= 2:
            car0, car1 = world_vehicle[0], world_vehicle[1]
            distance = np.sqrt((car0[0]-car1[0])**2 + (car0[1]-car1[1])**2)
            if distance < distance_threshold:
                new_car = [(car0[0]+car1[0])/2, (car0[1]+car1[1])/2]
                world_vehicle[0] = new_car
                ax.scatter(new_car[0], new_car[1], 10, c='b', marker='*')
            print('car distance', distance)

        if len(world_person) >= 2:
            per0, per1 = world_person[0], world_person[1]
            distance = np.sqrt((per0[0]-per1[0])**2 + (per0[1]-per1[1])**2)
            if distance < distance_threshold:
                new_person = [(per0[0]+per1[0])/2, (per0[1]+per1[1])/2]
                world_person[0] = new_person
                ax.scatter(new_person[0], new_person[1], 10, c='b', marker='*')
            print('person distance', distance)
        print('frame_time:', time.time()-start)

        # 写入统计
        if not world_vehicle:
            print('此帧没有发现车辆')
            world_vehicle.append([0, 0])
        if not world_person:
            print('此帧没有发现行人')
            world_person.append([0, 0])

        sum_car_value.append([int(world_vehicle[0][0]), int(world_vehicle[0][1])])
        sum_person_value.append([int(world_person[0][0]), int(world_person[0][1])])
        format = '{' + str(int(world_vehicle[0][0])) + ', ' + str(int(world_vehicle[0][1])) + '}, '
        sum_car += format
        format = '{' + str(int(world_person[0][0])) + ', ' + str(int(world_person[0][1])) + '}, '
        sum_person += format
        if index % 10 == 0 and not index == 0:
            sum_car += '\n'
            sum_person += '\n'

        if index == 5:
            sum_car += '}'
            sum_person += '}'
            print('======================= vehicle ==========================')
            print(sum_car)
            print('======================= person ==========================')
            print(sum_person)

        # 绘制原始bbox图像
        # if show_bbox:
        #     w, h = front.shape[1], front.shape[0]
        #     sheet_flat = np.zeros([h * 2, w * 2, 3], np.uint8)
        #     # col 1
        #     sheet_flat[:h, :w, :] = front
        #     sheet_flat[h:, :w, :] = left
        #     # col 2
        #     sheet_flat[:h, w:, :] = back
        #     sheet_flat[h:, w:, :] = right
        #     sheet_flat = cv.cvtColor(sheet_flat, cv.COLOR_BGR2RGB)
        #     # plt.figure('image')
        #     plt.subplot(122)
        #     plt.imshow(sheet_flat)
        #     plt.axis('off')

        if index == 223:  #351:
            break
        plt.pause(50)
        # plt.show()
        index += 1
        print(index)
        start = time.time()
    # np.save('image/sum_car1.npy', sum_car_value)
    # np.save('image/sum_person1.npy', sum_person_value)
    print('保存数据')
