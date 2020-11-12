import cv2
import glob
import os
import numpy as np


def convert(folder):
    tummor = ['benign', 'malignant']
    data = []
    x = 0

    for i in tummor:
        img_dir = folder + "/" + i
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = img.reshape(1, 784)
            img = np.append(img, [x])
            data.append(img)
        x = x + 1

    return data
