import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import h5py
from sklearn.metrics import confusion_matrix

# 定义 deep hashing learning 模型
class DHLNet(nn.Module):
    def __init__(self):
        super(DHLNet, self).__init__()
        roi = 90
        self.E2E = nn.Conv2d(1, 32, kernel_size=(roi, 1))
        self.E2N = nn.Conv2d(32, 64, kernel_size=(roi, 1))
        # self.E2N_DW = nn.Conv2d(1, 1, kernel_size=(roi, 1))
        # self.E2N_PW = nn.Conv2d(1, 64, kernel_size=(1, 1))
        self.N2G = nn.Conv2d(64, 128, kernel_size=(roi, 1))
        self.bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 96)
        self.hash = nn.Linear(96, 24)
        self.reg = nn.Linear(24, 8)
        self.fc2 = nn.Linear(96, 2)
        # self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.E2E(x) + self.E2E(x).transpose(3, 2))
        x = self.dropout(x)
        x = F.leaky_relu(self.E2N(x).transpose(3, 2) * 2)
        x = self.dropout(x)
        # x = self.E2N_DW(x).transpose(3, 2)*2
        # x = F.leaky_relu(self.E2N_PW(x))
        # x = self.dropout(x)
        x = F.leaky_relu(self.N2G(x))
        x = self.dropout(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        IF = F.leaky_relu(self.fc1(x))
        x = self.dropout(IF)
        x1 = self.hash(x)
        x2 = self.reg(x1)
        x3 = self.fc2(x)
        return IF, x1, x2, x3


def mapStd(X):
    preprocessor=prep.StandardScaler().fit(X)
    X = preprocessor.transform(X)
    return X


def mapMinmax(X):
    preprocessor=prep.MinMaxScaler().fit(X)
    X = 2*preprocessor.transform(X)-1
    return X

def sort_site():
    path = "D:/2 Projects/Data for BNC/ABIDE1-work3/data/"
    files = os.listdir(path)
    file_count = []

    for i, f in enumerate(files):
        file_list = os.listdir(path + f)
        file_count.append(len(file_list))

    # 使用zip()函数将站点名称和数据量进行配对
    site_data = list(zip(files, file_count))

    # 使用sorted()函数根据数据量对站点进行排序（按数据量从大到小）
    # key=lambda x: x[1]指定排序的依据是每个元素的第二个元素（即数据量），reverse=True表示按照从大到小的顺序排序。
    sorted_site_data = sorted(site_data, key=lambda x: (x[1], x[0]), reverse=False)

    # 输出排序后的站点名称和对应的数据量
    for site, count in sorted_site_data:
        print(f"Site: {site}, Data Count: {count}")

    return sorted_site_data


def load_data_():
    file = h5py.File('D:/2 Projects/Data for BNC/ABIDE1-work3/Tri_dataset/Tri_ASD_CALTECH.h5', 'r')
    dataset_net = file['tri_net']
    dataset_label = file['tri_label']

    # data = scio.loadmat('D:/2 Projects/Data for BNC/ABIDE1-work3/Tri_dataset/Tri_ASD_CALTECH.mat')

    X = dataset_net[:]

    Idx = [2, 3, 4, 5, 6, 7, 8, 9]  # 3:Age 4:Sex 5:Handedness 6:FIQ 7:VIQ 8:PIQ 9:EYE Status
    Y = dataset_label[:, Idx]

    file.close()

    col_idx = [1, 4, 5, 6]  # 3:Age 6:FIQ 7:VIQ 8:PIQ
    Y[:, col_idx] = mapStd(Y[:, col_idx])
    col_idx = [2, 3, 7]
    Y[:, col_idx] = mapMinmax(Y[:, col_idx])

    Y_d = Y[:, 0]

    roi = X.shape[1]

    ln = nn.LayerNorm(normalized_shape=[roi, roi], elementwise_affine=False)
    X = ln(torch.tensor(X)).view(-1, 1, roi, roi).type(torch.FloatTensor)

    Y = torch.tensor(Y)
    Y_d = torch.tensor(Y_d)


    return roi, X, Y, Y_d



def load_data(site):
    file = h5py.File('D:/2 Projects/Data for BNC/ABIDE1-work3/Tri_dataset/Tri_ASD_{}.h5'.format(site), 'r')
    X = file['tri_net'][:]
    Idx = [2, 3, 4, 5, 6, 7, 8, 9]  # 3:Age 4:Sex 5:Handedness 6:FIQ 7:VIQ 8:PIQ 9:EYE Status
    Y = file['tri_label'][:, Idx]

    # 关闭.h5文件
    file.close()

    col_idx = [1, 4, 5, 6]  # 3:Age 6:FIQ 7:VIQ 8:PIQ
    Y[:, col_idx] = mapStd(Y[:, col_idx])
    col_idx = [2, 3, 7]
    Y[:, col_idx] = mapMinmax(Y[:, col_idx])

    Y_d = Y[:, 0]

    roi = X.shape[1]

    ln = nn.LayerNorm(normalized_shape=[roi, roi], elementwise_affine=False)
    X = ln(torch.tensor(X)).view(-1, 1, roi, roi).type(torch.FloatTensor)

    Y = torch.tensor(Y)
    Y_d = torch.tensor(Y_d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    return roi, X, Y, Y_d


def comput_similarity(n, label):
    # n = batch_y.size(0)
    sim = torch.zeros([n, n]).type(torch.FloatTensor)
    for i in range(n):
        for j in range(n):
            if label[i, 0] == label[j, 0]:
                sim[i, j] = 1
            else:
                sim[i, j] = 0
    return sim


def train(train_loader, net, optimizer, ce_loss, reg_loss, sim_loss, triplet_loss):
    loss_sum = 0.

    train_hashcode = torch.empty(len(train_loader.dataset), 24)
    train_hashcode_y = torch.empty(len(train_loader.dataset))
    net.train()
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # sim = comput_similarity(batch_y.size(0), batch_y)

        optimizer.zero_grad()
        IF, out_hash, out_reg, out_fc = net(batch_x)

        label = batch_y[:, 0].long()

        # in_pro = torch.mm(out_hash, out_hash.transpose(1, 0))
        J_CE = ce_loss(out_fc, label)
        J_MES = reg_loss(out_reg, batch_y)
        J_TRI = triplet_loss(out_hash[0], out_hash[1], out_hash[2])
        loss_step = J_CE + J_MES + J_TRI  # 100 * ce_loss(out_fc, label) + 10 * sim_loss(in_pro, sim) + reg_loss(out_reg, batch_y) + triplet_loss(batch_x[0], batch_x[1], batch_x[2])
        hashcode = torch.div(torch.add(torch.sign(torch.sub(torch.sigmoid(out_hash), 0.5)), 1), 2)

        loss_step.backward(retain_graph=True)
        optimizer.step()

        loss_sum += loss_step.data.item()

        train_hashcode[step * train_loader.batch_size:step * train_loader.batch_size + len(batch_x), :] = hashcode
        train_hashcode_y[step * train_loader.batch_size:step * train_loader.batch_size + len(batch_x)] = batch_y[:, 0]

    loss = loss_sum / len(train_loader)

    return loss, train_hashcode, train_hashcode_y


def infer(loader, net, ce_loss, reg_loss, sim_loss):
    loss_sum = 0.
    test_hashcode = torch.empty(len(loader.dataset), 24)
    test_hashcode_y = torch.empty(len(loader.dataset))
    net.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):
            sim = comput_similarity(batch_y.size(0), batch_y)

            IF, out_hash, out_reg, out_fc = net(batch_x)

            label = batch_y[:, 0].long()

            in_pro = torch.mm(out_hash, out_hash.transpose(1, 0))
            loss_step = 100 * ce_loss(out_fc, label) + reg_loss(out_reg, batch_y)+triplet_loss(out_hash[0], out_hash[1], out_hash[2])
            hashcode = torch.div(torch.add(torch.sign(torch.sub(torch.sigmoid(out_hash), 0.5)), 1), 2)

            loss_sum += loss_step.data.item()


            test_hashcode[step * loader.batch_size:step * loader.batch_size + len(batch_x), :] = hashcode
            test_hashcode_y[step * loader.batch_size:step * loader.batch_size + len(batch_x)] = batch_y[:, 0]

        loss = loss_sum / len(loader)

    return loss, test_hashcode, test_hashcode_y

def comput_accuracy(output, target, train_y, y):
    y_ = torch.empty(y.shape[0])
    for i in range(y.shape[0]):
        y1 = output[i, :].type(torch.int)
        hm = y1 ^ target.type(torch.int)  # ^��ʾ���,Ҳ������ͬΪ0,��ͬΪ1
        dist = hm.sum(1)
        min = torch.min(dist)
        pos = []
        for k, x in enumerate(dist):
            if x == min:
                pos.append(k)
        label = []
        for t in range(len(pos)):
            label.append(train_y[pos[t]])
        if label.count(0) > label.count(1):  # >=
            y_[i] = 0
        else:
            y_[i] = 1
    correct_prediction = y.type(torch.int) ^ y_.type(torch.int)
    acc = 1 - sum(correct_prediction) / y.shape[0]
    CM = confusion_matrix(y, y_)
    tn, fp, fn, tp = CM.ravel()
    sen = tp / float((tp + fn))
    spe = tn / float((fp + tn))
    return acc.item(), sen, spe


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        loss = torch.clamp(self.margin + distance_positive - distance_negative, min=0.0).mean()
        return loss


if __name__ == "__main__":

    # ROI, X, Y, Y_d = load_data()

    BATCH_SIZE = 3
    EPOCH = 10#000

    ce_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss(reduction='mean')
    sim_loss = nn.BCEWithLogitsLoss(reduction='mean')
    # 定义间隔（margin）的值
    margin = 0.5
    # 实例化三元组损失函数
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    sorted_site_data = sort_site()

    for i, (test_site, _) in enumerate(sorted_site_data):
        print('test_site{}:{}'.format(i + 1, test_site))
        net = DHLNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        # 构建训练集和测试集
        train_sites = [site for site, _ in sorted_site_data if site != test_site]
        for j, site in enumerate(train_sites):
            ROI, X, Y, Y_d = load_data(site)
            train_dataset = Data.TensorDataset(X, Y.to(torch.float32))
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)  #会打乱顺序

            _, test_X, test_Y, test_Y_d = load_data(test_site)

            test_dataset = Data.TensorDataset(test_X, test_Y.to(torch.float32))
            test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            ax = []
            ay = []
            n_evaluation_epochs = 2
            plt.ion()

            for epoch in range(EPOCH):
                train_loss, train_hashcode, train_hashcode_y = train(train_loader, net, optimizer, ce_loss, reg_loss, sim_loss, triplet_loss)
                print("site:{} train_loss:{}".format(site,train_loss))

                test_loss, test_hashcode, test_hashcode_y = infer(test_loader, net, ce_loss, reg_loss, sim_loss)
                test_acc, sen, spe = comput_accuracy(test_hashcode, train_hashcode, train_hashcode_y, test_hashcode_y)
                print('test_loss = ', test_loss, 'test1_acc = ', test_acc, 'test1_sen = ', sen, 'test1_spe = ', spe)

                if epoch % n_evaluation_epochs == 0:
                    plt.figure(1)
                    ax.append(epoch + 1)
                    ay.append(np.mean(train_loss))
                    plt.clf()
                    plt.plot(ax, ay)
                    plt.pause(0.01)
                    plt.ioff()