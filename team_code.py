#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
from dataset_27cls_60s import *
from model_resnet34 import *

twelve_lead_model_filename = 'twelve_lead_best_model.pth'
six_lead_model_filename = 'six_lead_best_model.pth'
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
    train_num = int(num_recordings * 0.8)

    train_header_files = header_files[:train_num]
    vaild_header_files = header_files[train_num:]

    # 12 lead
    train_model(model_directory, train_header_files, vaild_header_files, twelve_leads)

    # 6 lead
    train_model(model_directory, train_header_files, vaild_header_files, six_leads)

    # 3 lead
    train_model(model_directory, train_header_files, vaild_header_files, three_leads)

    # 2 lead
    train_model(model_directory, train_header_files, vaild_header_files, two_leads)


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

        with torch.no_grad():
            logit = data_parallel(net, input) #net(input)
            probability = torch.sigmoid(logit)

            loss = F.binary_cross_entropy(probability, truth)

        valid_predict.append(probability.cpu().numpy())
        valid_truth.append(truth.cpu().numpy().astype(int))

        #---
        valid_loss += loss.cpu().numpy() * batch_size

        valid_num  += batch_size

        # print('\r %8d / %d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

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
def train_model(model_directory, train_header_files, vaild_header_files, leads):
    if len(leads) == 12:
        model_filename = twelve_lead_model_filename
        in_planes = 12
    elif len(leads) == 6:
        model_filename = six_lead_model_filename
        in_planes = 6
    elif len(leads) == 3 :
        model_filename = three_lead_model_filename
        in_planes = 3
    else:
        model_filename = two_lead_model_filename
        in_planes = 2

    batch_size = 8

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
        num_workers=0,
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
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (val_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_planes, num_classes=len(class_map)).cuda()

    # if initial_checkpoint is not None:
    #     state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    #
    #     net.load_state_dict(state_dict, strict=True)  # True

    log.write('net=%s\n' % (type(net)))
    log.write('\n')

    iter_accum = 1
    schduler = NullScheduler(lr=0.1)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0,
                                weight_decay=0.0)

    num_iters = len(train_dataset) * 2
    print(num_iters)
    iter_smooth = 400
    iter_log = 400
    iter_valid = 400
    iter_save = [0, num_iters - 1] \
                + list(range(0, num_iters, 400))  # 1*1000

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
            '%s   %0.5f %5.1f%s %4.1f | ' % (train_mode, rate, iter / 1000, asterisk, epoch,) + \
            '%4.3f | ' % loss + \
            '%0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | ' % (
                *precision_recall,) + \
            '%s' % (time_to_str((timer() - start_timer), 'min'))

        return text

    # ----
    top_loss = 99999
    train_loss = 0
    train_precision = [0 for i in range(len(class_map))]
    train_recall = [0 for i in range(len(class_map))]
    iter = 0
    i = 0

    start_timer = timer()
    log.write('num_iters' + ':' + str(num_iters))
    log.write('\n')
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

            # if 0:
            if (iter % iter_valid == 0):
                valid_loss, valid_precision, valid_recall = do_valid(net, valid_loader)  #
                pass

            if (iter % iter_log == 0):
                # print('\r', end='', flush=True)
                # print(message(rate, iter, epoch, train_loss, train_precision, train_recall, mode='log',
                #               train_mode='train'))
                log.write(str(iter) + ':' + str(valid_loss) + ' | time: %s min' % (time_to_str((timer() - start_timer))))

                log.write('\n')

            if valid_loss < top_loss:
                top_loss = valid_loss

                torch.save({
                    'iter': iter,
                    'epoch': epoch,
                }, model_directory + '/' + model_filename)
                if iter != start_iter:
                    torch.save(net.state_dict(), model_directory + '/' + model_filename)
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
            # print('\r', end='', flush=True)
            # print(message(rate, iter, epoch, batch_loss, batch_precision, batch_recall, mode='log', train_mode='train'),
            #       end='', flush=True)
            i = i + 1

        pass  # -- end of one data loader --
    pass  # -- end of all iterations --
    log.write('\n')

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + twelve_lead_model_filename
    net = Net(12, num_classes=len(class_map)).cuda()

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict,strict=False)
    net.load_state_dict(state_dict, strict=True)  # True
    return net

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + six_lead_model_filename
    net = Net(6, num_classes=len(class_map)).cuda()

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict,strict=False)
    net.load_state_dict(state_dict, strict=True)  # True
    return net

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + three_lead_model_filename
    net = Net(3, num_classes=len(class_map)).cuda()

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict,strict=False)
    net.load_state_dict(state_dict, strict=True)  # True
    return net

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    initial_checkpoint = model_directory + '/' + two_lead_model_filename
    net = Net(2, num_classes=len(class_map)).cuda()

    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    # net.load_state_dict(state_dict,strict=False)
    net.load_state_dict(state_dict, strict=True)  # True
    return net

# Generic function for loading a model.
# def load_model(filename):
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

    ecg = np.zeros((len(available_leads), 18000), dtype=np.float32)
    ecg[:, -temp_ecg.shape[1]:] = temp_ecg[:, -18000:]
    ecg = np.expand_dims(ecg, axis = 0)

    model.eval()
    input = torch.from_numpy(ecg).cuda()

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
    adc_gains = get_adcgains(header, leads)
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
    data_directory = DATA_ROOT_PATH + '/CinC2021/all_data'
    model_directory = DATA_ROOT_PATH + '/CinC2021_model'

    training_code(data_directory, model_directory)