#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
# from helper_code_old import *
import numpy as np, sys #, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from dataset_27cls_60s import *
# from model_resnet34 import *
from resnet import *
# from resnext import *
# from MobileNet import *
import scipy.io as sio  #read mat
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# ------------------add four lead ------------
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
# lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
# --------------------------------------------


twelve_lead_model_filename = 'twelve_lead_best_model.pth'
six_lead_model_filename = 'six_lead_best_model.pth'
four_lead_model_filename = 'four_lead_best_model.pth'
three_lead_model_filename = 'three_lead_best_model.pth'
two_lead_model_filename = 'two_lead_best_model.pth'

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    header_files = np.array(header_files)
    np.random.shuffle(header_files)
    train_num = int(num_recordings * 0.9)

    train_header_files = header_files[:train_num]
    vaild_header_files = header_files[train_num:]

    # 12 lead
    train_model(model_directory, train_header_files, vaild_header_files, twelve_leads, True)

    # # 6 lead
    train_model(model_directory, train_header_files, vaild_header_files, six_leads, True)
    #
    # # 4 lead
    train_model(model_directory, train_header_files, vaild_header_files, four_leads, True)
    # #
    # # # # 3 lead
    train_model(model_directory, train_header_files, vaild_header_files, three_leads, True)
    # #
    # # # 2 lead
    train_model(model_directory, train_header_files, vaild_header_files, two_leads, True)


def metric(truth, predict):
    truth_for_cls = np.sum(truth, axis=0) + 1e-11
    predict_for_cls = np.sum(predict, axis=0) + 1e-11

    # TP
    count = truth + predict
    count[count != 2] = 0
    TP = np.sum(count, axis=0) / 2

    precision = TP / predict_for_cls
    recall = TP / truth_for_cls

    return precision, recall

#------------------------------------
def do_valid(net, valid_loader):
    valid_loss = 0
    valid_predict = []
    valid_truth = []
    infors = []
    valid_num = 0

    for t, (input, truth, infor) in enumerate(valid_loader):
        batch_size = len(infor)

        infors.append(infor)

        net.eval()
        input  = input.cuda()
        truth  = truth.cuda()
        # print('\n input:', input.shape)

        with torch.no_grad():
            logit = data_parallel(net, input) #net(input)
            probability = torch.sigmoid(logit)

            loss = F.binary_cross_entropy(probability, truth)

        valid_predict.append(probability.cpu().numpy())
        valid_truth.append(truth.cpu().numpy().astype(int))

        #---
        valid_loss += loss.cpu().numpy() * batch_size

        valid_num  += batch_size
        # print('valid_num', valid_num)


        print('\r %8d / %d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --

    # assert(valid_num == len(valid_loader.dataset))
    valid_loss = valid_loss / (valid_num+1e-8)

    infors = np.hstack(infors)
    valid_truth = np.vstack(valid_truth)
    valid_predict = np.vstack(valid_predict)
    valid_predict_class = valid_predict>0.5

    valid_predict_class[:, -3] = 0

    valid_precision, valid_recall = metric(valid_truth, valid_predict_class.astype(int))

    return valid_loss, valid_precision, valid_recall

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
# def save_model(filename, classes, leads, imputer, classifier):
#     # Construct a data structure for the model and save it.
#     d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
#     joblib.dump(d, filename, protocol=0)

# train model
def train_model(model_directory, train_header_files, vaild_header_files, leads, pretrained):

    if len(leads) == 12:
        model_filename = twelve_lead_model_filename
        in_planes = 12
        num_iters = 8000  # 100000 #100000  # 3000000    #3000
        print('num_iters = %d\n' % (num_iters))
        net = resnet50(in_planes).cuda()

    elif len(leads) == 6:
        model_filename = six_lead_model_filename
        in_planes = 6
        num_iters = 8000 #6000
        print('num_iters = %d\n' % (num_iters))
        net = resnet50(in_planes).cuda()

    elif len(leads) == 4:
        model_filename = four_lead_model_filename
        in_planes = 4
        num_iters = 8000 #6000   6000
        print('num_iters = %d\n' % (num_iters))
        net = resnet50(in_planes).cuda()

    elif len(leads) == 3:
        model_filename = three_lead_model_filename
        in_planes = 3
        num_iters = 8000 #6000  6000
        print('num_iters = %d\n' % (num_iters))
        net = resnet50(in_planes).cuda()

    else:
        model_filename = two_lead_model_filename
        in_planes = 2
        num_iters = 8000 #6000  600
        print('num_iters = %d\n' % (num_iters))
        net = resnet50(in_planes).cuda()

    if pretrained:
        print("Loading pre-trained model.........")
        model_weight_path = "./model/twelve_lead_best_model.pth"
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        pre_weights = torch.load(model_weight_path, map_location="cuda")
        # # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

######################## batch_size ############################
    batch_size = 16 # 16  #8
    # ---
    Train_loss = []
    Train_recall = []


    out_dir = '.'
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = CinCDataset(
        mode = 'train',
        header_files=train_header_files,
        leads = leads
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=20,
        pin_memory=True,
        collate_fn=null_collate
    )

    val_dataset = CinCDataset(
        mode='train',
        header_files=vaild_header_files,
        leads = leads
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=10,
        pin_memory=True,
        collate_fn=null_collate
    )

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    log.write('batch_size = %d\n' % (batch_size))
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (val_dataset))
    log.write('\n')

    ## net -------------------------------------------------------------------
    log.write('** net setting **\n')
    # print('aaa')
    # print('in_planes:', in_planes)
    # net = Net(in_planes, num_classes=len(class_map)).cuda()   # ResNet34
    # net = resnet50(in_planes).cuda()    # ResNet50
    # net = resnext50(in_planes).cuda()    #new ResNeXt50
    # net = MobileNetV3_Large(in_planes, 27).cuda()  # MobileNetV3_Large


    log.write('net=%s\n' % (type(net)))
    log.write('\n')

    iter_accum = 1
    schduler = NullScheduler(lr=0.01)    #lr=0.1
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0,weight_decay=0)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.99, weight_decay=0.0005)   #weight_decay--L2 regularization

    # num_iters = 100000 # 80000   # 100000  #3000000    #3000
    iter_smooth = 400  # 400
    iter_log = 400  # 400
    iter_valid = 400  # 400
    iter_save = [0, num_iters - 1] \
                + list(range(0, num_iters, 400))  # 0

    start_iter = 0
    start_epoch = 0
    rate = 0

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('schduler\n  %s\n' % (schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n' % (batch_size, iter_accum))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write(
        '----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    log.write(
        'mode    rate    iter  epoch | loss  | 270492004 | 164889003 | 164890007 | 426627000 | 713427006 | 713426002 | 445118002 | 39732003  | 164909002 | 251146004 | 698252002 | 10370003  | 284470004 | 427172004 | 164947007 | 111975006 | 164917005 | 47665007  | 59118001  | 427393009 | 426177001 | 426783006 | 427084000 | 63593006  | 164934002 | 59931005  | 17338001  | time        \n')
    log.write(
        '----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    # train  0.01000   0.5   0.2 | 1.11  | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 0 hr 05 min
    def message(rate, iter, epoch, loss, precision, recall, mode='print', train_mode='train'):
        precision_recall = []
        for p, r in zip(precision, recall):
            precision_recall.append(p)
            precision_recall.append(r)

        if mode == ('print'):
            asterisk = ' '
        if mode == ('log'):
            asterisk = '*' if iter in iter_save else ' '

        text = \
            '%s   %0.3f %5.1f%s %4.1f | ' % (train_mode, rate, iter / 1000, asterisk, epoch,) + \
            '%4.3f | ' % loss + \
            '%0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | ' % (*precision_recall,) + \
            '%s' % (time_to_str((timer() - start_timer), 'min'))

        return text

    # ----
    top_loss = 99999
    train_loss = 0
    train_precision = [0 for i in range(len(class_map))]
    train_recall = [0 for i in range(len(class_map))]
    iter = 0
    i = 0

    # ---
    # Train_loss = []
    Valid_loss = []
    Valid_recall = []


    start_timer = timer()
    while iter < num_iters:
        train_predict_list = []
        train_truth_list = []
        sum_train_loss = 0
        sum_train = 0

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):
            batch_size = len(infor)
            iter = i + start_iter
            epoch = (iter - start_iter) * batch_size / len(train_dataset) + start_epoch
            # print('11111111')

            # if 0:
            if (iter % iter_valid == 0):
                valid_loss, valid_precision, valid_recall = do_valid(net, valid_loader)  #
                # print('222222222')
                pass

            # print('3333')

            if (iter % iter_log == 0):
                print('\r', end='', flush=True)
                print(message(rate, iter, epoch, train_loss, train_precision, train_recall, mode='log',
                              train_mode='train'))
                log.write(message(rate, iter, epoch, valid_loss, valid_precision, valid_recall, mode='log',
                                  train_mode='valid'))
                log.write('\n')


            # ==== save valid loss

            # Valid_loss.append(valid_loss)
            # sio.savemat('Valid_loss.mat', {'Valid_loss': Valid_loss})
            print('valid_iter=', iter)


            if valid_loss < top_loss:
                top_loss = valid_loss

                torch.save({
                    'iter': iter,
                    'epoch': epoch,
                }, model_directory + '/' + model_filename)
                print('iter=%d' % iter)  ######modify
                print('start_iter=%d' % start_iter)  ######modify
                if iter != start_iter:
                    print('iter=%d' % iter)
                    print('start_iter=%d' % start_iter)  ######modify
                    torch.save(net.state_dict(), model_directory + '/' + model_filename, _use_new_zipfile_serialization=False)
                    pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr < 0: break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)
            net.train()
            input = input.cuda()
            truth = truth.cuda()

            logit = data_parallel(net, input)
            probability = torch.sigmoid(logit)

            loss = F.binary_cross_entropy(probability, truth)

            loss.backward()
            loss = loss.detach().cpu().numpy()
            if (iter % iter_accum) == 0:
                optimizer.step()
                optimizer.zero_grad()

            predict = probability.cpu().detach().numpy()
            truth = truth.cpu().numpy().astype(int)
            batch_precision, batch_recall = metric(truth, (predict > 0.5).astype(int))

            # print statistics  --------
            batch_loss = loss
            train_predict_list.append(predict)
            train_truth_list.append(truth)
            sum_train_loss += loss * batch_size
            sum_train += batch_size
            if iter % iter_smooth == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                train_predict_list = np.vstack(train_predict_list)
                train_truth_list = np.vstack(train_truth_list)
                train_precision, train_recall = metric(train_truth_list, (train_predict_list > 0.5).astype(int))

                train_predict_list = []
                train_truth_list = []
                sum_train_loss = 0
                sum_train = 0


            # print(batch_loss)
            print('\r', end='', flush=True)
            print(message(rate, iter, epoch, batch_loss, batch_precision, batch_recall, mode='log', train_mode='train'),
                  end='', flush=True)
            i = i + 1

            # ---------save train loss
            print('train_iter=', iter)
            # Train_loss.append(batch_loss)
            # sio.savemat('Train_loss.mat', {'Train_loss': Train_loss})


        pass  # -- end of one data loader --
    pass  # -- end of all iterations --
    log.write('\n')



# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + twelve_lead_model_filename
    # net = Net(12, num_classes=len(class_map)).cuda()   #  resnet34
    net = resnet50(12).cuda()    #  resnet50
    # net = resnext50(12).cuda()    #new ResNeXt50
    # net = MobileNetV3_Large(12, 27).cuda()

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict, strict=False)
    net.load_state_dict(state_dict, strict=True)
    return net

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + six_lead_model_filename
    # net = Net(6, num_classes=len(class_map)).cuda()
    net = resnet50(6).cuda()  # resnet50

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict, strict=False)
    net.load_state_dict(state_dict, strict=True)
    return net

# Load your trained 4-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_four_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + four_lead_model_filename
    # net = Net(4, num_classes=len(class_map)).cuda()
    net = resnet50(4).cuda()  # resnet50

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict, strict=False)
    net.load_state_dict(state_dict, strict=True)
    return net


# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + three_lead_model_filename
    # net = Net(3, num_classes=len(class_map)).cuda()
    net = resnet50(3).cuda()  # resnet50

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict,strict=False)
    net.load_state_dict(state_dict, strict=True)
    return net

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + two_lead_model_filename
    # net = Net(2, num_classes=len(class_map)).cuda()
    net = resnet50(2).cuda()  # resnet50

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict, strict=False)
    net.load_state_dict(state_dict, strict=True)
    return net

# Generic function for loading a model.
# def load_model(filename):     #  original
#     return joblib.load(filename)
#
# def load_model(filename, leads):
#
#     return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording, twelve_leads)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording, six_leads)

# Run your trained 4-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_four_lead_model(model, header, recording):
    return run_model(model, header, recording, four_leads)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording, three_leads)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording, two_leads)

# Generic function for running a trained model.
def run_model(model, header, recording, leads):
    available_leads = get_leads(header)
    feature_indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        feature_indices.append(i)

    sampr = get_frequency(header)

    recording = np.array(recording / 1000)
    temp_ecg = []
    for i in feature_indices:
        temp_ecg.append(resample(recording[i, :], sampr))

    temp_ecg = np.array(temp_ecg)

    # ecg = np.zeros((len(available_leads), 18000), dtype=np.float32)
    ecg = np.zeros((len(leads), 18000), dtype=np.float32)
    ecg[:, -temp_ecg.shape[1]:] = temp_ecg[:, -18000:]
    ecg = np.expand_dims(ecg, axis = 0)

    model.eval()
    input = torch.from_numpy(ecg).cuda()
    # input = torch.from_numpy(ecg)  #modify==lym

    with torch.no_grad():
        logit = data_parallel(model, input)  # net(input)
        probability = torch.sigmoid(logit).cpu().numpy()[0]

    return class_map, (probability > 0.5).astype(int), probability

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms

if __name__ == '__main__':
    # data_directory = DATA_ROOT_PATH + '/CinC2021/all_data'
    data_directory = DATA_ROOT_PATH + '/CinC2021/split_data/train' #train'
    model_directory = DATA_ROOT_PATH + '/CinC2021_model'

    training_code(data_directory, model_directory)