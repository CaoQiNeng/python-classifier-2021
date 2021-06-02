#!/usr/bin/env python

# Do *not* edit this script.

import sys
from team_code import training_code
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


if __name__ == '__main__':
    # Parse arguments.
    # running in linux==================lym
    # if len(sys.argv) != 3:
    #     raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')
    #
    # data_directory = sys.argv[1]
    # model_directory = sys.argv[2]



    # data_directory = '/home/xhx/lym/CinC2021/data/CinC2021/all_data' # "/home/xhx/lym/CinC2021/data/CinC2020dataset/PhysioNetChallenge2020_Training_2/Training_2  "
    data_directory = '/home/xhx/lym/CinC2021/data/CinC2021/split_data/train'  #train
    model_directory = "/home/xhx/lym/CinC2021/python-classifier-2021/model_code"

    # #root in windows==lym
    # DATA_PATH = 'D:\Mywork\Eng\data\CinC2020dataset\PhysioNetChallenge2020_Training_2\Training_2'
    # Model_PATH = 'D:\Mywork\Eng\python-classifier-2021\model_code'
    # data_directory = DATA_PATH  #data root==lym
    # model_directory = Model_PATH #model root==lym

    # Run the training code.
    # try:
    #     print('Running training code...')
    #     training_code(data_directory, model_directory)
    # except:
    #     print('error')

    print('Running training code...')
    training_code(data_directory, model_directory)

    print('Done.')
