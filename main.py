# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import numpy as np
#
# def wh_iou(wh1, wh2):
#     # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
#     wh1 = wh1[:, None]  # [N,1,2]
#     wh2 = wh2[None]  # [1,M,2]
#     inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
#     return inter / (wh1.prod(2) + wh2.prod(2) - inter)
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     k=3
#     boxes = np.random.rand(38, 145, 90)
#     box_number = boxes.shape[0]
#     last_nearest = np.zeros((box_number,))
#     clusters = boxes[np.random.choice(box_number, k, replace=False)]
#     while True:
#         # 计算每个bboxes离每个簇的距离 1-IOU(bboxes, anchors)
#         distances = 1 - wh_iou(boxes, clusters)
#         # 计算每个bboxes距离最近的簇中心
#         current_nearest = np.argmin(distances, axis=1)
#         # 每个簇中元素不在发生变化说明以及聚类完毕
#         if (last_nearest == current_nearest).all():
#             break  # clusters won't change
#         for cluster in range(k):
#             # 根据每个簇中的bboxes重新计算簇中心
#             clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
#
#         last_nearest = current_nearest
#     print(clusters.shape)


import math
import random
from sklearn import datasets
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
try:
    from sklearn.datasets import make_blobs
except ImportError:
    from sklearn.datasets import make_blobs

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

def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    ran = random.choice(data_set)
    cluster_centers.append(ran) # 随机返回一个数据记录
    cluster_centers_p = []
    cluster_centers_p.append(data_set.index(ran))
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
            cluster_centers_p.append(j)
            break

        # cluster_centers.append(data_set[d.index(max(d))])

    return cluster_centers, cluster_centers_p


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
        # print(cluster_assment)
        for cent in range(k):  # 更新质心，将每个族中的点的均值作为质心
             pts_in_cluster = data[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
             centroids[cent, :] = np.mean(pts_in_cluster, axis=0)
    return centroids, cluster_assment
# 原文链接：https: // blog.csdn.net / weixin_48167570 / article / details / 122783739


if __name__ == "__main__":
    path = "D:/2 Projects/Data for BNC/ABIDE1-work3/CALTECH/"
    files = os.listdir(path)
    n = 90
    row = len(scio.loadmat(path + files[1])["ROISignals"])
    lie = row * n


    X = np.zeros((len(files), lie))
    X1 = np.zeros((len(files), row, n))
    for i, f in enumerate(files):
        data = scio.loadmat(path + f)["ROISignals"]
        data1 = data[0:row, 0:n]
        data2 = data1.flatten()
        X[i, :] = data2
        X1[i, :] = data1
    print(X.shape)

    optimalK = OptimalK(parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=np.arange(1, np.int32(len(X) * 0.1) + 1))
    print('Optimal number of clusters: ', n_clusters)

    kmeans = KMeans(n_clusters, init='k-means++', random_state=None, algorithm='auto', max_iter = 300 , tol = 0.0001).fit(X)

    print(kmeans.cluster_centers_)
    print(kmeans.labels_)

    data = X1  # 读取数据
    cluster_centers, cluster_centers_p = kpp_centers(data.tolist(), n_clusters)
    print('Initial cluster centers: ', cluster_centers_p)

    centroids, cluster_assment = kmeans(data.tolist(),n_clusters,cluster_centers)


    print(cluster_assment)



