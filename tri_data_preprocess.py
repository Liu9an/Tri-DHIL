import numpy as np
import os
import scipy.io as scio
import h5py
import pickle


if __name__ == "__main__":
    path = 'D:/2 Projects/Data for BNC/ABIDE1-work3/dataset/'  # ABIDE1-work3\ABIDE1160-work3\ABIDE1200-work3
    files = os.listdir(path)

    for _, f in enumerate(files):
        sitename = f[4:-4]
        net = scio.loadmat(path + f)["net"]
        fc = scio.loadmat(path + f)["fc"]
        label = scio.loadmat(path + f)["phenotype"]
        y = label[:, 2]
        cla, num = np.unique(y, return_counts=True)
        print("site:{} has {} cluster, every cluster contain {} subjects".format(sitename, len(cla), num))

        cluster_index = []
        anchor_index = []

        for i, c in enumerate(cla):
            n = num[i]
            row_index = np.where(y == i)[0]
            cluster_index.append(row_index)
            sub = net[row_index]

            # 计算数据之间的距离矩阵
            distances = np.zeros((n, n))
            for j in range(n):
                for k in range(n):
                    # 使用欧氏距离计算数据间的距离
                    distances[j, k] = np.linalg.norm(sub[j] - sub[k])

            # 选择距离最小的数据作为锚点
            index = np.argmin(np.sum(distances, axis=1))
            anchor_data = sub[index]
            anchor = row_index[index]
            anchor_index.append(anchor)

            # 打印选取的锚点和聚类数据
        #     print("class {}, anchor is {}".format(i, anchor))
        #     print("cluster:", row_index)
        # print("anchor_index:", anchor_index)
        # print("cluster_index:", cluster_index)

        triplets = []  # 存储三元组

        # 遍历每个类别
        for i, ci in enumerate(cluster_index):
            anchor = anchor_index[i]  # 当前类别的锚点

            # 遍历同类数据
            for positive in ci:
                if positive != anchor:
                    # 遍历不同类数据
                    for j, other_ci in enumerate(cluster_index):
                        if j != i:
                            for negative in other_ci:
                                # 构造三元组
                                triplet = [anchor, positive, negative]
                                triplets.append(triplet)


        # 打印三元组
        print("{} triplets".format(len(triplets)))
        # print("{} triplets are {}:".format(len(triplets), triplets))
        w_path = "D:/2 Projects/Data for BNC/ABIDE1-work3/Tri-list/{}_tri-list.txt".format(sitename)
        # 打开文件以写入模式（'w' 表示写入）
        with open(w_path.format(sitename), 'w') as file:
            for item in triplets:
                file.write(str(item) + '\n')
        '''
        tri_net = np.zeros((len(triplets)*3, net.shape[-2], net.shape[-1]))
        tri_label = np.zeros((len(triplets)*3, 9))

        # 构造三元组数据集
        for i, tri in enumerate(triplets):
            for j, t in enumerate(tri):
                tri_net[i*3+j, :, :] = net[t,:,:]
                tri_label[i*3+j, :] = label[t, :]
        '''


        # data = {
        #     'tri_net': tri_net,
        #     'tri_label': tri_label
        # }

        # 指定保存的文件路径和文件名
        # save_path = 'D:/2 Projects/Data for BNC/ADHD-work3/Tri_dataset/Tri_ADHD_{}.h5'.format(sitename)
        # file = h5py.File(save_path, 'w')
        # dataset_net = file.create_dataset('tri_net', data=tri_net)
        # dataset_label = file.create_dataset('tri_label', data=tri_label, dtype='float32')
        # file.close()
        # scio.savemat(save_path, data)