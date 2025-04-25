import numpy as np
import os
import scipy.io as scio
import pandas as pd




if __name__ == "__main__":
    path = "D:/2 Projects/Data for BNC/ABIDE2-work3/data/"  # ABIDE1-work3\ABIDE1160-work3\ABIDE1200-work3
    o_files = os.listdir(path)
    for oi, of in enumerate(o_files):
        print(of)
        files = os.listdir(path + of)
        roi = 90  # 90\160\200
        row = float('inf')
        for f in files:
            r = len(scio.loadmat(path + of + '/' + f)["ROISignals"])
            if r < row:
                row = r
        lie = row * roi
        M = pd.read_csv('D:/2 Projects/Data for BNC/ABIDE2-work3/Phenotypic.csv', header=None).values



        subjNum = len(files)
        ts_flat = np.zeros((subjNum, lie))
        ts = np.zeros((subjNum, row, roi))
        fc = np.zeros((subjNum, 4005))  # 4005\12720\19900
        net = np.zeros((subjNum, roi, roi))
        phenotype = np.zeros((subjNum, 10))

        for i, f in enumerate(files):
            data = scio.loadmat(path + of + '/' + f)["ROISignals"]
            data1 = data[0:row, 0:roi]
            data2 = data1.flatten()
            ts_flat[i, :] = data2
            ts[i, :] = data1


            subjname = int(f[-9:-4])  # 90:f[-22:-16]\160:f[-32:-25]\200:f[-25:-18]
            net_tmp = np.corrcoef(data1, rowvar=False)
            sum_nan = np.sum(np.isnan(net_tmp))

            if sum_nan == 0:
                net[i, :, :] = net_tmp
                fc[i, :] = net_tmp.flatten()[:4005]  # 4005\12720\19900
                row_index = np.where(M == subjname)[0]
                phenotype[i, :] = M[row_index, :]
                print('subject {}-{}'.format(i, subjname))
            else:
                print('subject {} has NaN'.format(subjname))

        data = {
            'ts': ts,
            'ts_flat': ts_flat,
            'net': net,
            'fc': fc,
            'phenotype': phenotype
        }

        # 指定保存的文件路径和文件名
        # 创建文件夹（如果不存在）
        folder_path = 'D:/2 Projects/Data for BNC/ABIDE2-work3/dataset'  # ABIDE1-work3\ABIDE1160-work3\ABIDE1200-work3
        os.makedirs(folder_path, exist_ok=True)
        file_name = 'ASD_{}.mat'.format(of.upper())
        file_path = os.path.join(folder_path, file_name)

        # 使用savemat函数保存数据
        scio.savemat(file_path, data)

