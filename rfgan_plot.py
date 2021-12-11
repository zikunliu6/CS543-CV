import matplotlib.pyplot as plt
import numpy as np
import os
# save_path = '../CV_real_data'
# save_path = '../CV_generated_data'
save_path = '../CV_generated_data_DC'

num_files = len([name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))])

num_plot = [1, 500, 1000, 1500, 1700]
for i in num_plot:
    plt.figure(figsize=(8, 5))
    data = np.load(save_path+'/{}.npy'.format(i))
    plt.imshow(data)
    plt.show()