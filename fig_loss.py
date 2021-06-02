import scipy.io as sio   #read mat
import numpy as np
import matplotlib.pyplot as plt
import random
plt.switch_backend('Agg')

mat_path = 'D:\Mywork\Eng\python-classifier-2021\mat'


Train_loss = sio.loadmat(mat_path + '/Train_loss.mat')['Train_loss']   #(1,16618)
# Train_recall = sio.loadmat(mat_path + '/Train_recall.mat')['Train_recall']
Valid_loss = sio.loadmat(mat_path + '/Valid_loss.mat')['Valid_loss']
# Valid_recall = sio.loadmat(mat_path + '/Valid_recall.mat')['Valid_recall']

# M, N = np.shape(Train_loss)

x1 = [i+1 for i in range(16618)]
y1 = Train_loss[0]



plt.figure(1)
plt.plot(x1, y1)
plt.xlabel("iter")#横坐标：频率
plt.ylabel("train-loss")#纵坐标：幅值
plt.show

x2 = [i+1 for i in range(16619)]
y2 = Valid_loss[0]
plt.figure(2)
plt.plot(x2, y2)
plt.xlabel("iter")#横坐标：频率
plt.ylabel("valid-loss")#纵坐标：幅值
plt.show










