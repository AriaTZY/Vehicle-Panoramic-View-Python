import numpy as np
import matplotlib.pyplot as plt
import time

# 应该是输入一组行人或是汽车的轨迹数据，然后使用卡尔曼滤波对路径进行滤波

car_sum = np.load('image/sum_car1.npy')
person_sum = np.load('image/sum_person1.npy')
assert len(car_sum) == len(person_sum), 'length is not equal'

car_x = []
car_y = []
person_x = []
person_y = []

pred_car_x = []
pred_car_y = []
pred_person_x = []
pred_person_y = []

# 0, 73    74, 153    154, 223    224, 351
iters = 0
predict_next_frame_x = 0
predict_next_frame_y = 0

predict_next_frame_x_person = 0
predict_next_frame_y_person = 0

print('共{}帧数据'.format(len(car_sum)))

for i in range(0, len(car_sum)):
    start = time.time()
    car_x.append(car_sum[i][0])
    car_y.append(car_sum[i][1])
    person_x.append(person_sum[i][0])
    person_y.append(person_sum[i][1])
    # 在没有足够多的积累点时首先先copy数据
    pred_car_x.append(car_sum[i][0])
    pred_car_y.append(car_sum[i][1])
    pred_person_x.append(person_sum[i][0])
    pred_person_y.append(person_sum[i][1])
    if iters >= 5:
        """  汽车的数据滤波部分  """
        distance = np.sqrt((predict_next_frame_x-car_x[-1])**2 + (predict_next_frame_y-car_y[-1])**2)
        # distance越大说明汽车运动的越不准确，需要减少weight，增加对预测值的信任
        if distance < 70:
            weight = 0.5
        else:
            weight = 0.1
        mix_data_x = weight * car_x[-1] + (1-weight) * predict_next_frame_x
        mix_data_y = weight * car_y[-1] + (1 - weight) * predict_next_frame_y
        pred_car_x[-1] = mix_data_x
        pred_car_y[-1] = mix_data_y
        """  行人的数据滤波  """
        distance = np.sqrt((predict_next_frame_x_person - person_x[-1]) ** 2 + (predict_next_frame_y_person - person_y[-1]) ** 2)
        # distance越大说明汽车运动的越不准确，需要减少weight，增加对预测值的信任
        if distance < 60:
            weight = 0.5
        else:
            weight = 0.1
        mix_data_x = weight * person_x[-1] + (1 - weight) * predict_next_frame_x_person
        mix_data_y = weight * person_y[-1] + (1 - weight) * predict_next_frame_y_person
        pred_person_x[-1] = mix_data_x
        pred_person_y[-1] = mix_data_y
    # 数据预测
    if iters >= 4:
        """  汽车预测  """
        avr_x_delta = 0
        avr_y_delta = 0
        for j in range(3):
            avr_x_delta += pred_car_x[iters-j] - pred_car_x[iters-j-1]
            avr_y_delta += pred_car_y[iters - j] - pred_car_y[iters - j - 1]
        avr_x_delta /= 3
        avr_y_delta /= 3
        predict_next_frame_x = pred_car_x[iters] + avr_x_delta
        predict_next_frame_y = pred_car_y[iters] + avr_y_delta
        """  行人预测  """
        avr_x_delta = 0
        avr_y_delta = 0
        for j in range(3):
            avr_x_delta += pred_person_x[iters - j] - pred_person_x[iters - j - 1]
            avr_y_delta += pred_person_y[iters - j] - pred_person_y[iters - j - 1]
        avr_x_delta /= 3
        avr_y_delta /= 3
        predict_next_frame_x_person = pred_person_x[iters] + avr_x_delta
        predict_next_frame_y_person = pred_person_y[iters] + avr_y_delta
        print('time interval:', time.time()-start)
    iters += 1

plt.plot(car_x, car_y, 'k<-')
plt.plot(person_x, person_y, 'bo-')

plt.plot(pred_car_x, pred_car_y, 'co-')
plt.plot(pred_person_x, pred_person_y, 'c>--')
plt.xlim([-600, 600])
plt.ylim([-900, 600])
plt.show()