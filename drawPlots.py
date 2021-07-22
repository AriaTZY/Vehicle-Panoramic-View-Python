import cv2 as cv
import numpy as np
import math
import random

# 这个是为了画出简单的点对点的映射，用来研究鱼眼相机的非线性映射的

# dist = np.array([-0.3359, 0.1285, -0.0231, 0])
dist = np.array([-0.02037657262403154, -0.014287103171377493, 0.025033228794991685, -0.013057747261212267])
# width = 600
# half = width//2
# # 绘制无畸变的线
# for angle in range(0, 180, 5):
#     sheet = np.zeros([width, width, 3], np.uint8)
#     # 绘制坐标轴
#     cv.line(sheet, (0, width // 2), (width, width // 2), (0, 255, 255), 3)
#     cv.line(sheet, (width // 2, 0), (width // 2, width), (0, 255, 255), 3)
#     # 绘制原始角度
#     startPoint = (width // 2, width // 2)
#     endPoint = (int(math.sin(math.radians(angle))*half), int(math.cos(math.radians(angle))*half))
#     endPoint = (endPoint[0] + half, -endPoint[1] + half)
#     cv.line(sheet, startPoint, endPoint, (255, 255, 255), 2)
#     cv.circle(sheet, endPoint, 5, (200, 200, 200), 2)
#     # 绘制去畸变后的角度
#     arcAngle = math.radians(angle)
#     undistoredAngle = arcAngle * (1 + dist[0] * arcAngle ** 2 + dist[1] * arcAngle ** 4 + dist[2] * arcAngle ** 6 + dist[3] * arcAngle ** 8)
#     endPoint2 = (int(math.sin(undistoredAngle) * half), int(math.cos(undistoredAngle) * half))
#     endPoint2 = (endPoint2[0] + half, -endPoint2[1] + half)
#     cv.line(sheet, startPoint, endPoint2, (40, 40, 255), 2)
#     cv.circle(sheet, endPoint2, 5, (20, 20, 200), 2)
#     print('origin', arcAngle*57.3, 'now', undistoredAngle*57.3)
#     cv.imshow("SHEET", sheet)
#     cv.waitKey(0)



# 绘制平面上的径向畸变的函数
width = 600
half = width//2
sheet = np.ones([width, width, 3], np.uint8)
sheet = sheet * 50
# 绘制坐标轴
cv.line(sheet, (0, width // 2), (width, width // 2), (255, 255, 255), 1)
cv.line(sheet, (width // 2, 0), (width // 2, width), (255, 255, 255), 1)
# 绘制无畸变的线
for var in range(0, 630, 30):
    for i in range(2):  # 这个是为了做对称
        # 得到初始的xyz
        o_x = var
        o_y = 30
        x = o_x - half
        y = o_y - half
        z = 300
        if i == 0:  # 对称
            y = -y
            o_y = 600-o_y
        x = x/z
        y = y/z
        r = math.sqrt(x**2 + y**2)
        print(x, y)
        theta = math.atan(r)
        undistoredAngle = theta * (1 + dist[0] * theta ** 2 + dist[1] * theta ** 4 + dist[2] * theta ** 6 + dist[3] * theta ** 8)
        xd = undistoredAngle / r * x
        yd = undistoredAngle / r * y
        u = int(xd*300 + half)
        v = int(yd*300 + half)

        cv.line(sheet, (o_x, o_y), (u, v), (100, 255, 100), 1)
        cv.circle(sheet, (o_x, o_y), 4, (0, 255, 250), -1)
        cv.circle(sheet, (u, v), 4, (200, 250, 12), -1)

cv.imshow("SHEET", sheet)
cv.waitKey(0)