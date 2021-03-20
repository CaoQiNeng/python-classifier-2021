from common import *
from scipy import signal
from helper_code import *

#--------------
class_map = pd.read_csv('/physionet/evaluation-2021/dx_mapping_scored.csv')['SNOMED CT Code'].to_numpy()
# class_map = pd.read_csv('/home1/cqn/data_root/CinC2021/evaluation-2021/dx_mapping_scored.csv')['SNOMED CT Code'].to_numpy()

class CinCDataset(Dataset):
    def __init__(self, mode, header_files, leads):
        self.mode = mode
        self.header_files = header_files

        self.num_image = len(self.header_files)
        self.leads = leads

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        string += '\tmode     = %s\n'%self.mode
        string += '\tnum_image = %d\n'%self.num_image
        return string

    def __len__(self):
        return self.num_image

    def __getitem__(self, index):
        header_file = self.header_files[index]

        header = load_header(header_file)

        sampr = get_frequency(header)

        old_temp_ecg = sio.loadmat(header_file.replace('hea', 'mat'))['val']
        old_temp_ecg = np.array(old_temp_ecg / 1000)
        temp_ecg = []
        feature_indices = [twelve_leads.index(lead) for lead in self.leads]
        for i in feature_indices:
            temp_ecg.append(resample(old_temp_ecg[i,:], sampr))

        temp_ecg = np.array(temp_ecg)

        ecg = np.zeros((len(self.leads), 18000), dtype=np.float32)
        ecg[:,-temp_ecg.shape[1]:] = temp_ecg[:,-18000:]

        infor = Struct(
            index  = index,
            header_file = header_file,
        )

        label = np.zeros(len(class_map))
        if self.mode == 'train':
            labels = get_labels(header)

            for l in labels:
                l_index = np.where(class_map == int(l))[0]
                if len(l_index) > 0:
                    label[l_index] = 1

        return ecg, label, infor

class CustomSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = self.dataset.First_label

        self.AF_index  = np.where(label == 0)[0]
        self.I_AVB_index = np.where(label == 1)[0]
        self.LBBB_index = np.where(label == 2)[0]
        self.Normal_index = np.where(label == 3)[0]
        self.PAC_index = np.where(label == 4)[0]
        self.PVC_index = np.where(label == 5)[0]
        self.RBBB_index = np.where(label == 6)[0]
        self.STD_index = np.where(label == 7)[0]
        self.STE_index = np.where(label == 8)[0]

        #assume we know neg is majority class
        num_RBBB = len(self.RBBB_index)
        self.length = 9 * num_RBBB

    def __iter__(self):
        RBBB = self.RBBB_index.copy()
        np.random.shuffle(RBBB)
        # num_RBBB = len(self.RBBB_index)

        AF = np.random.choice(self.AF_index, len(self.AF_index), replace=False)
        I_AVB = np.random.choice(self.I_AVB_index, len(self.I_AVB_index), replace=False)
        LBBB = np.random.choice(self.LBBB_index, len(self.PAC_index), replace=True)
        Normal = np.random.choice(self.Normal_index, len(self.Normal_index), replace=False)
        PAC = np.random.choice(self.PAC_index, len(self.PAC_index), replace=True)
        PVC = np.random.choice(self.PVC_index, len(self.PVC_index), replace=False)
        STD = np.random.choice(self.STD_index, len(self.PAC_index), replace=False)
        STE = np.random.choice(self.STE_index, len(self.STD_index), replace=True)

        l = np.hstack([AF,I_AVB,LBBB,Normal,PAC,PVC,RBBB,STD,STE])
        np.random.shuffle(l)

        return iter(l)

    def __len__(self):
        return self.length

class BalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = self.dataset.First_label

        self.AF_index  = np.where(label == 0)[0]
        self.I_AVB_index = np.where(label == 1)[0]
        self.LBBB_index = np.where(label == 2)[0]
        self.Normal_index = np.where(label == 3)[0]
        self.PAC_index = np.where(label == 4)[0]
        self.PVC_index = np.where(label == 5)[0]
        self.RBBB_index = np.where(label == 6)[0]
        self.STD_index = np.where(label == 7)[0]
        self.STE_index = np.where(label == 8)[0]

        #assume we know neg is majority class
        num_RBBB = len(self.RBBB_index)
        self.length = 9 * num_RBBB

    def __iter__(self):
        RBBB = self.RBBB_index.copy()
        np.random.shuffle(RBBB)
        num_RBBB = len(self.RBBB_index)

        AF = np.random.choice(self.AF_index, num_RBBB, replace=True)
        I_AVB = np.random.choice(self.I_AVB_index, num_RBBB, replace=True)
        LBBB = np.random.choice(self.LBBB_index, num_RBBB, replace=True)
        Normal = np.random.choice(self.Normal_index, num_RBBB, replace=True)
        PAC = np.random.choice(self.PAC_index, num_RBBB, replace=True)
        PVC = np.random.choice(self.PVC_index, num_RBBB, replace=True)
        STD = np.random.choice(self.STD_index, num_RBBB, replace=True)
        STE = np.random.choice(self.STE_index, num_RBBB, replace=True)

        l = np.stack([AF,I_AVB,LBBB,Normal,PAC,PVC,RBBB,STD,STE]).T
        l = l.reshape(-1)

        return iter(l)

    def __len__(self):
        return self.length

def resample(data, sampr, after_Hz=300):
    data_len = len(data)
    propessed_data = signal.resample(data, int(data_len * (after_Hz / sampr)))

    return propessed_data

def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    label = torch.from_numpy(np.stack(label)).float()
    input = torch.from_numpy(np.stack(input)).float()

    return input, label, infor

def run_check_DataSet():
    data_directory = DATA_DIR + '/all_data'
    header_files, recording_files = find_challenge_files(data_directory)

    header_files = np.array(header_files)
    np.random.shuffle(header_files)
    train_num = int(len(header_files) * 0.8)

    train_header_files = header_files[:train_num]

    train_dataset = CinCDataset(
        header_files=train_header_files,
        leads=three_leads
    )

    a = 0
    for t, (input, truth, infor) in enumerate(train_dataset):
        a += 1

        if np.sum(truth) > 0 :
            print(infor.ecg_id)
            print(truth)



# main #################################################################
if __name__ == '__main__':
    run_check_DataSet()

    print('\nsucess!')