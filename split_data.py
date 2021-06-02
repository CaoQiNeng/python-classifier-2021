import os
from shutil import copy
import random
from tqdm import tqdm
# file_path = 'D:\Mywork\Eng\data\CinC2021/all_data'
# train_path = r'D:\Mywork\Eng\data\CinC2021\split_data\train'
# test_path = r'D:\Mywork\Eng\data\CinC2021\split_data\test'

file_path = '/home/xhx/lym/CinC2021/data/CinC2021/all_data'
train_path = '/home/xhx/lym/CinC2021/data/CinC2021/split_data/train'
test_path = '/home/xhx/lym/CinC2021/data/CinC2021/split_data/test'

datas = os.listdir(file_path)
test_size = 0.1
all_data_name = []
for data in datas:
    data_split = data.split('.')
    if data_split[1] == "mat":
        all_data_name.append(data_split[0])

num = len(all_data_name)
print(num)
random.shuffle(all_data_name)
trainnum = num - int(num * test_size)
train_data_name = all_data_name[:trainnum]
test_data_name = all_data_name[trainnum:]
for file_name in tqdm(train_data_name):
    mat_file = os.path.join(file_path,file_name + '.mat')
    hea_file = os.path.join(file_path,file_name + '.hea')
    copy(mat_file,os.path.join(train_path,file_name + '.mat'))
    copy(hea_file,os.path.join(train_path,file_name + '.hea'))
for file_name in tqdm(test_data_name):
    mat_file = os.path.join(file_path, file_name + '.mat')
    hea_file = os.path.join(file_path, file_name + '.hea')
    copy(mat_file, os.path.join(test_path, file_name + '.mat'))
    copy(hea_file, os.path.join(test_path, file_name + '.hea'))