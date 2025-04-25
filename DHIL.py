
import os
import math
import random
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
try:
    from sklearn.datasets import make_blobs
except ImportError:
    from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(random.choice(data_set)) # 随机返回一个数据记录
    d = [0 for _ in range(len(data_set))]  # 初始化d为0，长度为数据集的长度
    p = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i], p[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()  # 5、取得sum_all之间的随机值
        for j, di in enumerate(d):  # 6、获得距离最远的样本点作为聚类中心点
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[j])
            break
    return cluster_centers

def euler_distance(point1, point2) :
    """
    计算两点之间的欧拉距离，支持多维
    """
    a = np.array(point1)
    b = np.array(point2)
    distance = np.sum(np.linalg.norm(a - b, axis=0))
    return distance

def get_closest_dist(point, centroids):
    p = 0
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(point, centroid)
        if dist < min_dist:
            min_dist = dist
            p = i
    return min_dist, p

def kmeans(data, k, cent):
    '''
    kmeans算法求解聚类中心
    :param data: 训练数据
    :param k: 聚类中心的个数
    :param cent: 随机初始化的聚类中心
    :return: 返回训练完成的聚类中心和每个样本所属的类别
    '''
    m = np.shape(data)[0]  # 获得行数m
    cluster_assment = np.mat(np.zeros((m, 2)))  # 初试化一个矩阵，用来记录簇索引和存储距离平方
    centroids = cent  # 生成初始化点
    cluster_changed = True  # 判断是否需要重新计算聚类中心
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            distance_min = np.inf  # 设置样本与聚类中心之间的最小的距离，初始值为正无穷
            index_min = -1  # 所属的类别
            for j in range(k):
                distance_ji = euler_distance(centroids[j], data[i])
                if distance_ji < distance_min:
                    distance_min = distance_ji
                    index_min = j
            if cluster_assment[i, 0] != index_min:
                cluster_changed = True
                cluster_assment[i, :] = index_min, distance_min ** 2  # 存储距离平方
    return centroids, cluster_assment


if __name__ == '__main__':
    path = "D:/2 Projects/Data for BNC/ABIDE1-work3/NYU/"
    files = os.listdir(path)
    n = 90
    row = len(scio.loadmat(path+files[1])["ROISignals"])
    lie = row * n

    X = np.zeros((len(files), lie))
    X1 = np.zeros((len(files), row, n))
    for i,f in enumerate(files):
        data = scio.loadmat(path+f)["ROISignals"]
        data1 = data[0:row, 0:n]
        data2 = data1.flatten()
        X[i, :] = data2
        X1[i, :] = data1
    print(X.shape)

    optimalK = OptimalK(parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=np.arange(1, np.int32(len(X)*0.15)+1))
    print('Optimal clusters: ', n_clusters)
    # print(optimalK.gap_df.head())
    #
    # plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
    # plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
    #             optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
    # plt.grid(True)
    # plt.xlabel('Cluster Count')
    # plt.ylabel('Gap Value')
    # plt.title('Gap Values by Cluster Count')
    # plt.show()

    k = n_clusters
    # X1_number = X1.shape[0]
    # last_nearest = np.zeros((X1_number,))
    # clusters = X1[np.random.choice(X1_number, k, replace=False)]
    #
    # while True:
    #     distances = np.zeros((X1_number, k))
    #     for i, x in enumerate(X1):
    #         for j, c in enumerate(clusters):
    #             distances[i, j] = np.sum(np.linalg.norm(x - c, axis=0))
    #     current_nearest = np.argmin(distances, axis=1)
    #     if (last_nearest == current_nearest).all():
    #         break
    #     for cluster in range(k):
    #         # 根据每个簇中的bboxes重新计算簇中心
    #         clusters[cluster] = np.median(X1[current_nearest == cluster], axis=0)
    #
    #     last_nearest = current_nearest

    data = X1  # 读取数据
    cluster_centers = kpp_centers(data.tolist(), k)
    centroids, cluster_assment = kmeans(data.tolist(), k, cluster_centers)

    print(cluster_assment)

