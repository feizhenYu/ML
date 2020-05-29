import pandas as pd
import numpy as np
import math
import csv

data = pd.read_csv('train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
data = data.to_numpy()

x = np.zeros([12 * 471, 9 * 18], dtype=float)
y = np.zeros([12 * 471, 1], dtype=float)

month_data = {}
for month in range(12):
    sample = np.zeros([18, 20*24], dtype=float)
    for day in range(20):
        sample[:, 24*day: 24*(day+1)] = data[18*(month*20+day): 18*(month*20+day+1), :]
    month_data[month] = sample

for month in range(12):
    for hour in range(471):
        x[hour+month*471, :] = month_data[month][:, hour:hour+9].reshape(1, -1)
        y[hour+month*471, 0] = month_data[month][9, hour+9]

mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(471*12):
    for j in range(18*9):
        if std_x[j] != 0:
            x[i][j] = (x[i][j]-mean_x[j])/std_x[j]

dim = 18*9*2+1
w = np.zeros([dim, 1])
x = np.concatenate((np.power(x, 2), x), axis=1).astype(float)
x = np.concatenate((np.ones([471*12, 1]), x), axis=1).astype(float)
x_train = x[:math.floor(len(x)*0.8), :]
y_train = y[:math.floor(len(x)*0.8), :]
x_validation = x[math.floor(len(x)*0.8):, :]
y_validation = y[math.floor(len(x)*0.8):, :]
adagrad = np.zeros([dim, 1])
learning_rate = 10
eps = 0.000000001
iter_time = 20000
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2))/math.floor(471*12*0.8))
    if t % 1000 == 0:
        print(t, ":", loss)
    gradient = 2 * np.dot(np.transpose(x_train), np.dot(x_train, w)-y_train)
    adagrad += np.power(gradient, 2)
    w = w - learning_rate*gradient/np.sqrt(adagrad+eps)
np.save('weight.npy', w)
print('---------------------')

validation_loss = np.sqrt(np.power(np.dot(x_validation, w) - y_validation, 2))
v_loss = np.sqrt(np.power(np.dot(x_validation, w) - y_validation, 2))/(y_validation+0.00000000000000000000000000000001)
with open('validation_result.csv', mode='w', newline='') as result_file:
    csv_writer = csv.writer(result_file)
    header = ['loss']
    print(header)
    csv_writer.writerow(header)
    # for i in range(len(y_validation)):
    #     if y_validation[i] != 0:
    #         csv_writer.writerow(validation_loss[i] / y_validation[i])
    #     else:
    #         csv_writer.writerow(validation_loss[i])
    csv_writer.writerows(v_loss)
print('---------------------')

test_data = pd.read_csv('test.csv', header=None, encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
x_test = np.zeros([240, 18*9])
for i in range(240):
    x_test[i, :] = test_data[18*i: 18*(i+1), :].reshape(1, -1)
# test_mean = np.mean(x_test, axis=0)
# test_std = np.std(x_test, axis=0)
for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        if std_x[j] != 0:
            x_test[i][j] = (x_test[i][j]-mean_x[j])/std_x[j]
x_test = np.concatenate((np.power(x_test, 2), x_test), axis=1).astype(float)
x_test = np.concatenate((np.ones([240, 1]), x_test), axis=1).astype(float)

w = np.load('weight.npy')
result = np.dot(x_test, w)

with open('submit_file.csv', mode='w', newline='') as submit_f:
    csv_writer = csv.writer(submit_f)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for index in range(240):
        row = ['id_'+str(index), result[index, 0]]
        csv_writer.writerow(row)
