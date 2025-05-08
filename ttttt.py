import mne
import numpy as np
import scipy.io as sio
cnt_file = './test.cnt'
channel_num = 66
first_label = 2
non_eeg_chn = [65,66]

mat=np.load('./data.npz')
data_revice = mat['data']

print(data_revice)

print(data_revice[-1])

print(np.max(data_revice[-1]))

print(np.sort(data_revice[-1]))


# sio.savemat('C:/Users/11273/data.mat', {'data': data_revice})
index = data_revice[-1, :]-65280
idx = np.argwhere(index == first_label)
data_r = data_revice[:channel_num, idx[0, 0]:(idx[0, 0]+1000*3)]
# data_r_n = data_r/5033.0

raw = mne.io.read_raw_cnt(cnt_file, preload=True)
event, event_dict = mne.events_from_annotations(raw)
event_idx = event[0, 0]
data_save = raw[:,:][0]
data_r_cnt = data_save[:, event_idx:(event_idx+1000*3)]
div_ = data_r[:,:-1]/data_r_cnt[:,1:]
print(data_revice)
div_eeg = 0
for i in range(channel_num):
    if i+1 not in non_eeg_chn:
        div_eeg += np.mean(div_[i,:])
div_eeg /= channel_num-len(non_eeg_chn)
print('div_eeg = ' + str(div_eeg))

print('Done')