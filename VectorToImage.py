import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./Hiragana/train_data.csv', sep=',')

data = data.iloc[:, 2:].values
for i in range(10):
    image = data[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.show()