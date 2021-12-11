# CS543-CV

This is the repo for cs543 computer vision final project

**Dependency:**

```
pytorch
numpy
scipy
matplotlib
```

**File Explanation**

1. rfgan.py contains the GAN structure and training function. It will automatically save the trained model into a newly built folder.
2. rfgan_test.py is used to generate large amount of heatmaps using the trained model.
3. rfgan_tunining.py is used for parameter tuning such as learning rate and batch size, which will show a generated heatmap every epoch to let you monitor the training process.

- To run the training, do following:

```
python main.py
```

- The dataset contains over 6000 heatmaps that was collected by us using FMCW Radar. 
- The dataset can be found here:

```
https://drive.google.com/file/d/1KX_JeYDH9c5amwbS4e5kUv-gdSO6VwMQ/view?usp=sharing
```

Each .mat file in the folder means a heatmap, which has the dimension of 80 by 180, representing distance and angle respectively.

