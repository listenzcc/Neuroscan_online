import time
import numpy as np
# from pynput import keyboard
import threading

from myBuffer import Buffer
from parameters import *

# from Algorithm.MyAlgorithm import MyAlgorithm
# from Algorithm.MyAlertor import MyAlertor
from Algorithm.Feature_extraction_202505 import EEG_Feature_Algorithm, EYE_Feature_Algorithm
from Algorithm.Attention_Vigilance_Workload_predictor_202505 import Attention_Vigilance_Workload_Algorithm, State_Acc_Regression_Algorithm
from FCM_module import fuzzy_inference_test

switch = True

div = 5033.1648/1e6
t = 1  # 每过多少秒调用一次算法
windowLength = 4#60  # 每次获取的数据长度（秒）

state_predictor = Attention_Vigilance_Workload_Algorithm(step=windowLength)
acc_predictor = State_Acc_Regression_Algorithm(step=windowLength)
EEG_feature_extraction_algorithm = EEG_Feature_Algorithm(step=windowLength)
EYE_feature_extraction_algorithm = EYE_Feature_Algorithm(step=windowLength)
# alertor = MyAlertor()

infos = {'connection_info': {'IP': server_host, 'port': str(server_port)}}
my_buffer = Buffer(nchan=channel_num, windowLength=windowLength, srate=1000)
my_buffer.on(infos['connection_info']['IP'], int(infos['connection_info']['port']))
my_buffer.start()

# key_thread.start()
predicts = np.zeros((0, 3))
data_save = np.zeros((channel_num + 1, 0))
cur_time = time.time()
# for i in range(data_r_cnt.shape[1]//4000):
#     data = data_r_cnt[:, i*4000:(i+1)*4000]
#     data = np.concatenate([data, np.zeros((1, 4000))], axis=0)
time.sleep(2)
i=0
while switch:
    while time.time() - cur_time < t:
        continue
    cur_time = time.time()
    # print(cur_time)
    data = my_buffer.output()
    data = data/div
    # data_save = np.concatenate([data_save, data], axis=1)

    # EEG_data:
    EEG_features, EOG_features = EEG_feature_extraction_algorithm.feature_extraction(data[:65, :])

    # EYE_data: 'Gaze point X', 'Gaze point Y', 'Pupil diameter left', 'Pupil diameter right', 'Location'
    EYE_features = EYE_feature_extraction_algorithm.feature_extraction(data[65:, :])

    # Input_features = np.concatenate([EEG_features, EOG_features, EYE_features], axis=1)
    attention_predict_label, vigilance_predict_label, workload_predict_label = state_predictor.algorithm(EEG_features, EOG_features, EYE_features)
    acc_predict_label = acc_predictor.algorithm(attention_predict_label, vigilance_predict_label, workload_predict_label)

    # input: attention, vigilance, workload, acc, 环境，任务紧急程度
    environmental_condition = 1
    task_urgency = 1
    speed = fuzzy_inference_test.run([attention_predict_label, vigilance_predict_label, workload_predict_label, acc_predict_label, environmental_condition, task_urgency])

    # alertor.alarm(i, result)
    # result = np.concatenate([result, data[-1:,:1], np.array([[cur_time]])], axis=1)
    # predicts = np.concatenate([predicts, result], axis=0)
    # i += 1

my_buffer.stop()
my_buffer.off()
np.savez('./predict', data=predicts)
# np.savez('./data_6', data=data_save)


# from matplotlib import pyplot as plt
# p_cnt = np.load('predict1.npz')
# p_cnt = p_cnt['data']
# plt.plot(p_cnt)
# plt.show()

# p_ol=np.load('C:/Users/HaibaoWang/Desktop/exp_data/abbreviated_vigilance/wkn/predict5.npz')
# p_ol=p_ol['data']
# plt.plot(p_ol)
# plt.show()

# raw_data = pop_loadcnt(data_path, 't1', 0);
# labels_location = [raw_data.event.latency].';
# eeg_data = raw_data.data(:, labels_location(1):labels_location(end));
