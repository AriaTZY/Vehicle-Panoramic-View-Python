import cv2
import numpy as np
import glob


# =-=-=-=-=-=-=-=下面的是通过opencv进行的相机标定数据-=-=-=-=-=-=-=-=-=-=-
# K = np.array([[238.87052313877663, 0.0, 328.1263268763279],
#               [0.0, 236.36510731401265, 228.35450912275593],
#               [0.0, 0.0, 1.0]])
# D = np.array([-0.03231958237996581, -0.0022693899191978772, 0.008768319032577323, -0.0073711397484963566])
# DIM = (640, 480)

RT_front = np.array([[-0.9940,    0.0224,    0.0345 ,   1.4697],
   [-0.0156,    0.5618,   -0.8291, -278.1589],
   [-0.0383,   -0.8333,   -0.5581,  103.8625]])


# 引入K，D以及DIM，顺序为前-左-后-右
def init_K_D():
    K = np.zeros((4, 3, 3), np.float)
    D = np.zeros((4, 4), np.float)
    DIM = (640, 480)
    # 写入front数据
    K[0] = np.array([[225.72178081263067, 0.0, 321.96633799109753], [0.0, 223.2285288966986, 246.3869556120706], [0.0, 0.0, 1.0]])
    D[0] = np.array([-0.02037657262403154, -0.014287103171377493, 0.025033228794991685, -0.013057747261212267])
    # 写入left数据
    K[1] = np.array([[218.65531794403827, 0.0, 310.50121174602526], [0.0, 215.39957902366683, 247.30826877876999], [0.0, 0.0, 1.0]])
    D[1] = np.array([-0.014172403622643947, -0.03822754173759392, 0.034992974105021085, -0.011792807943546112])
    # 写入back数据
    K[2] = np.array([[209.69847738130434, 0.0, 331.567127128282], [0.0, 207.3995501674224, 228.76814345973017], [0.0, 0.0, 1.0]])
    # D[2] = np.array([-0.01925745462786066, -0.02134484872200362, 0.024590932749542434, -0.008861872434744875])
    D[2] = np.array([-0.01925745462786066, 0.02134484872200362, -0.024590932749542434, 0.008861872434744875])
    # 写入right数据
    K[3] = np.array([[212.1669546072108, 0.0, 323.11293903148265], [0.0, 210.04749072390408, 239.1898811865337], [0.0, 0.0, 1.0]])
    D[3] = np.array([-0.018228219102242156, -0.011667407425982976, 0.021229884587139515, -0.008917534350805383])
    return K, D, DIM


def cali_live(pic):
    # 相机内参，从matlab中直接获取，他的f/dx写在一起了作为了一个参数
    K = np.array([[240.13, 0, 329],
                  [0, 237.63, 229],
                  [0, 0,      1]])
    D = np.array([-0.3359, 0.1285, -0.0231, 0])
    shaped = pic.shape
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.identity(3), K, (shaped[1], shaped[0]), cv2.CV_32FC1)
    dst = cv2.remap(pic, map1, map2, cv2.INTER_CUBIC)
    # alpha是0保留最小区域，为1还会有黑边，但会保留所有像素，其实这一步不用也行
    # new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (shaped[1], shaped[0]), alpha=0)
    # dst = cv.fisheye.undistortImage(pic, mtx, dist)
    # dst = cv.undistort(pic, mtx, dist, None, new_mtx)
    return dst


def undistort(img_path, param):
    DIM, K, D = param
    img = cv2.imread(img_path)
    img = cv2.resize(img, DIM)
    rvecs = np.zeros((3, 1), dtype=np.float64)
    tvecs = np.zeros((3, 1), dtype=np.float64)
    world_point = np.zeros([4, 1], dtype=np.float)

    for z in range(0, 900, 30):
        # z = (1 / 150.0) * (y-600)**2
        world_point[0] = -300
        world_point[1] = 500
        world_point[2] = z
        world_point[3] = 1
        point = np.dot(RT_front, world_point)
        point = np.reshape(point,  [1, 1, 3])
        print('camera point', point)
        new_mtx, ROI = cv2.getOptimalNewCameraMatrix(K, D, DIM, 1, DIM)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_mtx, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        distoredPoint, jacobian = cv2.fisheye.projectPoints(point, rvecs, tvecs, K, D)
        center = (int(distoredPoint[0][0][0]), int(distoredPoint[0][0][1]))
        # cv2.circle(img, center, 5, (255, 100, 3), 2)
        # cv2.imshow('distored', img)
        # cv2.waitKey(0)
    # 测试undistored函数
    distored_uv = np.array([[[100, 300]]], np.float)
    # print(distored_uv.shape)
    # cv2.circle(img, (distored_uv[0][0][0], distored_uv[0][0][1]), 5, (255, 100, 3), 2)
    undistored_uv = cv2.fisheye.undistortPoints(distored_uv, K, D)
    print(undistored_uv)
    undistored_xyz = np.array([[[undistored_uv[0][0][0], undistored_uv[0][0][1], 1]]])
    distoredxyz, jacobian = cv2.fisheye.projectPoints(undistored_xyz, rvecs, tvecs, K, D)
    print(distoredxyz)
    distoredPoint, jacobian = cv2.fisheye.projectPoints(point, rvecs, tvecs, K, D)
    cv2.imshow('distored', img)
    cv2.imshow('undistored', undistorted_img)
    cv2.waitKey(0)
    # cv2.imwrite('E:\\WORKPLACE\\3DSurround\\pictures.png', undistorted_img)


def get_K_and_D(checkerboard, imgsPath):
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    # print('objp is:', objp)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            print('特征点检测成功')
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
            for i in range(54):
                cv2.circle(img, (corners[i][0][0], corners[i][0][1]), 3, (0, 255, 255), 2)
            cv2.imshow('deg', img)
            cv2.waitKey(10)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, rvecs, tvecs = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    return DIM, K, D


if __name__ == '__main__':
    # DIM, K, D = get_K_and_D((6, 9), "../pictures/calibration/back")
    K, D, DIM = init_K_D()
    K = K[0]
    D = D[0]
    # undistort("E:/WORKPLACE/3DSurround/pictures/joint/1_right.jpg", [DIM, K[3], D[3]])
    undistort("E:/WORKPLACE/3DSurround/pictures/joint/1_front.jpg", [DIM, K, D])



