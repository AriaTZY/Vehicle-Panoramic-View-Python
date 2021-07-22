import copy
import cv2 as cv


CV_KEY_S = 115
CV_KEY_A = 97

# 这是进行可用摄像头测试
# cameraList = []
# for i in range(9):
#     cap = cv.VideoCapture(i)
#     success1, _ = cap.read()
#     if success1:
#         cameraList.append(i)
# print("可用摄像头索引：", cameraList)


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
    if event == cv.EVENT_LBUTTONUP:
        print('[', x, ',', y, '],')


index = 0


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
    sheetCenter = (sheet.shape[1]//2, sheet.shape[0]//2)
    sheet[15:15+h, sheetCenter[0]-w//2:sheetCenter[0]+w//2] = pic1
    sheet[sheetCenter[1] - w // 2:sheetCenter[1] + w // 2, \
          10:10+h] = pic2
    sheet[sheet.shape[0] - 15 - h:sheet.shape[0] - 15, \
          sheetCenter[0] - w // 2:sheetCenter[0] + w // 2] = pic3
    sheet[sheetCenter[1] - w // 2:sheetCenter[1] + w // 2, \
          sheet.shape[1] - 10 - h:sheet.shape[1] - 10] = pic4
    sheet = sheet[15:-15, :, :]  # emmmm当时留多了
    sheet = cv.resize(sheet, (0, 0), None, 0.85, 0.85)
    cv.imshow('carmodel', sheet)
    get = cv.waitKey(10)
    if get == CV_KEY_S:
        save_name = '../pictures/joint/' + str(index) + '_front.jpg'
        cv.imwrite(save_name, o_pic1)
        save_name = '../pictures/joint/' + str(index) + '_left.jpg'
        cv.imwrite(save_name, o_pic2)
        save_name = '../pictures/joint/' + str(index) + '_back.jpg'
        cv.imwrite(save_name, o_pic3)
        save_name = '../pictures/joint/' + str(index) + '_right.jpg'
        cv.imwrite(save_name, o_pic4)
        print(index, " 存储完成")
        index += 1


if __name__ == '__main__':
    cfg = [1,2,3,4]
    cap1 = cv.VideoCapture(cfg[0])
    cap2 = cv.VideoCapture(cfg[1])
    cap3 = cv.VideoCapture(cfg[2])
    cap4 = cv.VideoCapture(cfg[3])
    while True:
        _, pic1 = cap1.read()
        _, pic2 = cap2.read()
        _, pic3 = cap3.read()
        _, pic4 = cap4.read()
        jointFourImages(pic1, pic2, pic3, pic4)

    index = 24
    while True:
        success1, pic1 = cap1.read()
        cv.imshow('TEST', pic1)
        cv.setMouseCallback('TEST', mousecallback)

        char = cv.waitKey(1)
        if char == CV_KEY_S:
            save_name = '../pictures/calibration/' + str(index) + '.jpg'
            cv.imwrite(save_name, pic1)
            index += 1
            print('保存成功第', index, '照片')
            cv.waitKey(500)





