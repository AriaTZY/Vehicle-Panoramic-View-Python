# 有关论文的东西，好像并没有什么卵用
import os
import cv2 as cv

root_path = 'image/picture_mode/back'
txt_root_path = 'image/picture_mode/txt/left'

count = 0


def mousecallback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        print(x, y)


for i in range(6):
    count = 0
    txt_name = os.path.join(txt_root_path, 'img_'+str(i)+'.txt')
    img_name = os.path.join(root_path, 'img_'+str(i)+'.jpg')
    print(txt_name)
    img = cv.imread(img_name)
    cv.imshow('P', img)
    cv.setMouseCallback('P', mousecallback)
    cv.waitKey(0)


