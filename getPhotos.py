import copy
import cv2 as cv
import numpy as np
import time

# 这是保存使用图像，进行USB摄像头与索引进行对应的最常用函数

# 这是进行可用摄像头测试
# cameraList = []
# for i in range(9):
#     cap = cv.VideoCapture(i)
#     success1, _ = cap.read()
#     if success1:
#         cameraList.append(i)
# print("可用摄像头索引：", cameraList)

CV_KEY_S = 115
CV_KEY_A = 97
cfg = [1,0,5,4]


# 把图像旋转一定角度并且裁剪它
def rotateImage(pic, angle):
    center = (pic.shape[1], 0)
    picShape = pic.shape
    if angle == 0:
        ret = pic
    elif angle == 90:
        M = cv.getRotationMatrix2D(center, 90, 1.0)
        pic = cv.warpAffine(pic, M, (pic.shape[0] * 3, pic.shape[1]))
        ret = pic[:, picShape[1]:(picShape[1] + picShape[0]), :]
    elif angle == 180:
        ret = cv.flip(pic, 0)
        ret = cv.flip(ret, 1)
    elif angle == -90:
        M = cv.getRotationMatrix2D(center, 90, 1.0)
        pic = cv.warpAffine(pic, M, (pic.shape[0] * 3, pic.shape[1]))
        ret = pic[:, picShape[1]:(picShape[1] + picShape[0]), :]
        ret = cv.flip(ret, 1)
        ret = cv.flip(ret, 0)
    return ret


# 预处理读入的原始图片，主要工作有裁剪、缩放、同时提供旋转角度
def preProcess(pic, angle):
    pic = pic[50:430, :, :]
    pic = cv.resize(pic, (0, 0), None, 0.5, 0.5)
    return rotateImage(pic, angle)


def mousecallback(event, x, y, flags, param):
    global cfg
    if event == cv.EVENT_LBUTTONUP:
        print(x, y)
        print('[', x, ',', y, '],')
        if y < 158:  # 前方摄像头
            print('camera index:', cfg[0])
        elif y > 435:  # 后方位置摄像头
            print('camera index:', cfg[2])
        else:
            if x < 167:  # 左方摄像头
                print('camera index:', cfg[1])
            elif x > 335:
                print('camera index:', cfg[3])


def jointFourImages(pic1, pic2, pic3, pic4):
    global index
    # pic1 = cv.imread('../pictures/calibration/0.jpg', 1)
    # pic2 = cv.imread('../pictures/calibration/1.jpg', 1)
    # pic3 = cv.imread('../pictures/calibration/2.jpg', 1)
    # pic4 = cv.imread('../pictures/calibration/3.jpg', 1)
    o_pic1 = copy.copy(pic1)  # 最原始的图像
    o_pic2 = copy.copy(pic2)
    o_pic3 = copy.copy(pic3)
    o_pic4 = copy.copy(pic4)

    pic1 = preProcess(pic1, 0)
    pic2 = preProcess(pic2, 90)
    pic3 = preProcess(pic3, 180)
    pic4 = preProcess(pic4, -90)

    # 创建画布
    # h, w, _ = pic1.shape
    # sheetWidth = 3 * h + 20  # 10是留的边框距
    # sheetHeight = h*2 + w + 30  # 纵向15的留边
    # sheet = np.zeros([sheetHeight, sheetWidth, 3], np.uint8)
    # carPic = cv.imread("E:\\WORKPLACE\\3DSurround\\pictures\\car model\\car1.jpg", 1)
    # carPic = cv.resize(carPic, (h, w))
    # sheet[15 + h:15 + h + w, 10 + h:10 + h + h, :] = carPic

    h, w, _ = pic1.shape
    sheet = cv.imread("E:\\WORKPLACE\\3DSurround\\pictures\\car model\\sheet1.jpg", 1)
    sheetCenter = (sheet.shape[1] // 2, sheet.shape[0] // 2)
    sheet[15:15 + h, sheetCenter[0] - w // 2:sheetCenter[0] + w // 2] = pic1
    sheet[sheetCenter[1] - w // 2:sheetCenter[1] + w // 2, \
    10:10 + h] = pic2
    sheet[sheet.shape[0] - 15 - h:sheet.shape[0] - 15, \
    sheetCenter[0] - w // 2:sheetCenter[0] + w // 2] = pic3
    sheet[sheetCenter[1] - w // 2:sheetCenter[1] + w // 2, \
    sheet.shape[1] - 10 - h:sheet.shape[1] - 10] = pic4
    sheet = sheet[15:-15, :, :]  # emmmm当时留多了
    sheet = cv.resize(sheet, (0, 0), None, 0.85, 0.85)
    cv.imshow('carmodel', sheet)
    cv.setMouseCallback('carmodel', mousecallback)
    get = cv.waitKey(10)
    if get == CV_KEY_S:
        save_name = '../pictures/joint/outdoors/' + str(index) + '_front.jpg'
        cv.imwrite(save_name, o_pic1)
        save_name = '../pictures/joint/outdoors/' + str(index) + '_left.jpg'
        cv.imwrite(save_name, o_pic2)
        save_name = '../pictures/joint/outdoors/' + str(index) + '_back.jpg'
        cv.imwrite(save_name, o_pic3)
        save_name = '../pictures/joint/outdoors/' + str(index) + '_right.jpg'
        cv.imwrite(save_name, o_pic4)
        print(index, " 存储完成")
        index += 1


def save_single_image():
    index = 40
    cap1 = cv.VideoCapture(1)
    while True:
        _, pic1 = cap1.read()
        cv.imshow('TEST', pic1)
        a = cv.waitKey(10)
        if a == CV_KEY_S:
            save_name = '../pictures/calibration/back/' + str(index) + '.jpg'
            cv.imwrite(save_name, pic1)
            print(index, 'done')
            index += 1


# 存储视频，以图像方式存储连帧视频
def save_videos(path, fps=20.):
    import os
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(path + '/front', exist_ok=True)
        os.makedirs(path + '/left', exist_ok=True)
        os.makedirs(path + '/back', exist_ok=True)
        os.makedirs(path + '/right', exist_ok=True)
    cap1 = cv.VideoCapture(cfg[0])
    cap2 = cv.VideoCapture(cfg[1])
    cap3 = cv.VideoCapture(cfg[2])
    cap4 = cv.VideoCapture(cfg[3])
    time_interval = int(1000/fps)
    index = 0
    save_count = 0
    save_flag = False
    while True:
        _, pic1 = cap1.read()  # 前
        _, pic2 = cap2.read()  # 左
        _, pic3 = cap3.read()  # 后
        _, pic4 = cap4.read()  # 右
        w, h = pic1.shape[1], pic1.shape[0]
        sheet = np.zeros([h*2, w*2, 3], np.uint8)
        # row 1
        sheet[:h, :w, :] = pic1
        sheet[h:, :w, :] = pic3
        # row 2
        sheet[:h, w:, :] = pic2
        sheet[h:, w:, :] = pic4
        cv.imshow('joint', cv.resize(sheet, (0, 0), None, 0.5, 0.5))
        get = cv.waitKey(int(time_interval))
        save_count += 1
        if get == CV_KEY_S:
            save_flag = not save_flag
            save_count = 0  # 保存计数清零
            print('是否开始保存:', save_flag)
        if save_flag and save_count == 5:
            save_flag = False
            print('保存图片！')
            file_name = os.path.join(path, 'front', 'img_' + str(index) + '.jpg')
            cv.imwrite(file_name, pic1)

            file_name = os.path.join(path, 'left', 'img_' + str(index) + '.jpg')
            cv.imwrite(file_name, pic2)

            file_name = os.path.join(path, 'back', 'img_' + str(index) + '.jpg')
            cv.imwrite(file_name, pic3)

            file_name = os.path.join(path, 'right', 'img_' + str(index) + '.jpg')
            cv.imwrite(file_name, pic4)

            file_name = os.path.join(path, 'sheet_' + str(index) + '.jpg')
            cv.imwrite(file_name, sheet)

            print(index)
            index += 1


start = 0
end = 0
if __name__ == '__main__':
    save_videos('image/2020_10_26/', fps=1)
    # 下面这个是用来做摄像头校正的
    index = 11
    cap1 = cv.VideoCapture(cfg[0])
    cap2 = cv.VideoCapture(cfg[1])
    cap3 = cv.VideoCapture(cfg[2])
    cap4 = cv.VideoCapture(cfg[3])
    start = time.time()
    index = 0
    while True:
        _, pic1 = cap1.read()
        _, pic2 = cap2.read()
        _, pic3 = cap3.read()
        _, pic4 = cap4.read()
        jointFourImages(pic1, pic2, pic3, pic4)
        index += 1
        # if time.time()-start > 1:
        #     print('帧率：', index)
        #     print('真实秒数：', time.time()-start)
        #     index = 0
        #     start = time.time()






