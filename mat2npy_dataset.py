import scipy
import numpy as np
import os
import utils

save_path = '../CV_training_data_old'
save_path2 = '../CV_real_data'
utils.mkr(save_path2)

num_files = len([name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))])

for i in range(num_files):
    trace = scipy.io.loadmat(save_path+'/{}.mat'.format(i+1))['mat']
    np.save(save_path2+'/{}.npy'.format(i), trace[1:, 1:])