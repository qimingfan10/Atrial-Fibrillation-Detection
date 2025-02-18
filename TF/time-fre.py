import os
import datetime
import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from PyEMD import EEMD, EMD, Visualisation
from scipy.signal import hilbert
from vmdpy import VMD
import time
from multiprocessing import Pool
from tqdm import tqdm

# 设置中文字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用 SimHei 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def fftlw(Fs, y, draw):
    L = len(y)
    f = np.arange(int(L / 2)) * Fs / L
    M = np.abs(fft(y)) * 2 / L
    M = M[0:int(L / 2)]
    M[0] = M[0] / 2
    return f, M

def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def getDataSet(number, X_data, Y_data):
    print("正在读取 " + number + " 号心电数据...")
    record_name = 'files/01'
    header = wfdb.rdheader(record_name)
    record = wfdb.rdrecord(f'files/{number}', channel_names=['ECG'])
    if record.p_signal is None:
        data = record.d_signal.flatten()
    else:
        data = record.p_signal.flatten()
    rdata = denoise(data=data)
    annotation = wfdb.rdann('files/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.aux_note
    ecgClassSet = ['(N', '(AFIB', '(AFL', '(J']
    n = len(annotation.symbol)
    N = len(record.p_signal)
    lable_sample = [0] * n
    for i in range(n):
        if Rclass[i] == '(AFIB':
            lable_sample[i] = 1
    lable = np.zeros(N)
    for i in range(0, n - 1):
        lable[Rlocation[i]:Rlocation[i + 1]] = lable_sample[i]
    lable[Rlocation[n - 1]:N] = lable_sample[n - 1]
    i = 0
    while (i + 250) < N:
        try:
            x_train = rdata[i:i + 250]
            X_data.append(x_train)
            Y_data.append(np.max(lable[i:i + 250]))
            i += 250
        except ValueError:
            i += 250
    non_af_num = Y_data.count(0)
    af_num = Y_data.count(1)
    all_num = Y_data.count(0) + Y_data.count(1)
    return non_af_num, af_num, all_num

def get_atr_file_names(folder_path):
    atr_file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.atr'):
            base_name = os.path.splitext(file_name)[0]
            atr_file_names.append(base_name)
    return atr_file_names

def loadData():
    numberSet = get_atr_file_names('files')
    dataSet = []
    lableSet = []
    non_af_num_list, af_num_list, all_num_list = [], [], []
    for n in numberSet:
        non_af_num, af_num, all_num = getDataSet(n, dataSet, lableSet)
        non_af_num_list.append(non_af_num)
        af_num_list.append(af_num)
        all_num_list.append(all_num)
    dataSet = np.array(dataSet).reshape(-1, 250)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    X = train_ds[:, :250]
    Y = train_ds[:, 250]
    n_n = 30
    num = Y.shape[0] // n_n
    X = X[0:num * n_n, :]
    Y = Y[0:num * n_n]
    X = X.reshape(-1, 250 * n_n)
    Y_new = []
    for i in range(0, len(Y), n_n):
        if (Y[i:i + n_n] == 1).any():
            Y_new.append(1)
        else:
            Y_new.append(0)
    Y = np.array(Y_new)
    RATIO = 0.3
    np.random.seed(12)
    shuffle_index = np.random.permutation(len(Y))
    test_index = shuffle_index[0:]
    X_, Y_ = X[test_index], Y[test_index]
    return X_, Y_

def decompose_lw(signal, t, method, K=5, draw=1):
    names = ['emd', 'eemd', 'vmd']
    idx = names.index(method)
    if idx == 0:
        emd = EMD()
        IMFs = emd.emd(signal)
    elif idx == 2:
        alpha = 2000
        tau = 0.
        DC = 0
        init = 1
        tol = 1e-7
        IMFs, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    else:
        eemd = EEMD()
        emd = eemd.EMD
        emd.extrema_detection = "parabol"
        IMFs = eemd.eemd(signal, t)
    return IMFs

def hhtlw(IMFs, t, f_range=[0, 40], t_range=[0, 1], ft_size=[128, 128], draw=1):
    fmin, fmax = f_range[0], f_range[1]
    tmin, tmax = t_range[0], t_range[1]
    fdim, tdim = ft_size[0], ft_size[1]
    dt = (tmax - tmin) / (tdim - 1)
    df = (fmax - fmin) / (fdim - 1)
    vis = Visualisation()
    c_matrix = np.zeros((fdim, tdim))
    for imf in IMFs:
        imf = np.array([imf])
        freqs = abs(vis._calc_inst_freq(imf, t, order=False, alpha=None))
        amp = abs(hilbert(imf))
        freqs = np.squeeze(freqs)
        amp = np.squeeze(amp)
        temp_matrix = np.zeros((fdim, tdim))
        n_matrix = np.zeros((fdim, tdim))
        for i, j, k in zip(t, freqs, amp):
            if i >= tmin and i <= tmax and j >= fmin and j <= fmax:
                temp_matrix[round((j - fmin) / df)][round((i - tmin) / dt)] += k
                n_matrix[round((j - fmin) / df)][round((i - tmin) / dt)] += 1
        n_matrix = n_matrix.reshape(-1)
        idx = np.where(n_matrix == 0)[0]
        n_matrix[idx] = 1
        n_matrix = n_matrix.reshape(fdim, tdim)
        temp_matrix = temp_matrix / n_matrix
        c_matrix += temp_matrix
    t = np.linspace(tmin, tmax, tdim)
    f = np.linspace(fmin, fmax, fdim)
    if draw == 1:
        fig, axes = plt.subplots()
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.contourf(t, f, c_matrix, cmap="jet")
        plt.axis('off')
    return t, f, c_matrix

def process_signal(args):
    i, signal, t, Fs, method, K, output_dir, prefix = args
    _, _ = fftlw(Fs, signal, 0)
    IMFs = decompose_lw(signal, t, method=method, K=K)
    tt, ff, c_matrix = hhtlw(IMFs, t, f_range=[0, 30], t_range=[0, 30], ft_size=[100, 100])
    plt.tight_layout(pad=1.0)
    image_path = os.path.join(output_dir, f"{prefix}_{i}.png")
    plt.savefig(image_path, dpi=300)
    plt.close()
    return image_path

def main():
    X, Y = loadData()
    X_0 = X[Y == 0, :]
    X_1 = X[Y == 1, :]
    n_n = 30

    tic = time.time()
    Fs = 250
    t = np.arange(0, n_n, 1 / 250)
    N_0 = Y[Y == 0].shape[0]
    N_1 = Y[Y == 1].shape[0]

    print("N_train_0", N_0)
    print("N_train_1", N_1)

    base_dir = r"D:\BaiduSyncdisk\社团与活动\心电图\nn5" #存储路径
    af_dir = os.path.join(base_dir, "AF")
    n_dir = os.path.join(base_dir, "N")

    os.makedirs(af_dir, exist_ok=True)
    os.makedirs(n_dir, exist_ok=True)

    # 并行处理 AF 信号
    af_args = [(i, X_1[i, :], t, Fs, 'vmd', 5, af_dir, "AF") for i in range(N_1)]
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(process_signal, af_args), total=N_1, desc="Processing AF signals"):
            pass

    # 并行处理 N 信号
    n_args = [(i, X_0[i, :], t, Fs, 'vmd', 5, n_dir, "N") for i in range(N_0)]
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(process_signal, n_args), total=N_0, desc="Processing N signals"):
            pass

    toc = time.time()
    print("For loop:" + str(1000 * (toc - tic)) + "ms")

if __name__ == '__main__':
    main()