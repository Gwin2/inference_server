from scipy.signal.windows import hamming
import numpy as np
import math

def pre_func_cross_corelation(data=None, window_size=8000):  # data - массив длиной 50 записей каждая по 400000
    data1=[]
    for i in range(len(data) - 1):
        signal1 = [math.sqrt(data[i][0][j]**2 + data[i][1][j]**2) for j in range(400000)]
        signal2 = [math.sqrt(data[i+1][0][j]**2 + data[i+1][1][j]**2) for j in range(400000)]


        x1 = signal1
        x2 = signal2


        hamming_window = hamming(window_size)

        downsampled_signal1 = np.zeros(len(x1) // window_size)
        downsampled_signal2 = downsampled_signal1.copy()


        for j in range(len(x1) // window_size):
            downsampled_signal1[j] = np.sum(x1[j * window_size:(j + 1) * window_size] * hamming_window)
            downsampled_signal2[j] = np.sum(x2[j * window_size:(j + 1) * window_size] * hamming_window)

        corr = np.correlate(downsampled_signal1, downsampled_signal2, 'same').max()

        data1.append(corr)
        print(data1)

    return data1


def inference_func_cross_corelation(data=None):
    data1 = data[:len(data) // 2]
    data2 = data[len(data) // 2:]

    weighted_var1 = np.var(data1) / np.mean(data1)
    weighted_var2 = np.var(data2) / np.mean(data2)
    print(weighted_var1, weighted_var2)
    if weighted_var2 / weighted_var1 < 0.25:
        alarm = 1
        result = 'ALARM'
    else:
        alarm = 0
        result = 'NEALARM'

    print('Инференс завершен')
    print()
    return alarm, result 


def post_func_cross_corelation(prediction=0):
    if prediction:
        print('ALARM!')
    else:
        print('NE ALARM')
        
    print('Постобработка завершена')
    print()
