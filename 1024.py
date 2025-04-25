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
import random


# 定义 deep hashing learning 模型
class DHLNet(nn.Module):
    def __init__(self):
        super(DHLNet, self).__init__()
        roi = 90
        self.E2E = nn.Conv2d(1, 32, kernel_size=(roi, 1))
        self.E2N = nn.Conv2d(32, 64, kernel_size=(roi, 1))
        self.N2G = nn.Conv2d(64, 128, kernel_size=(roi, 1))
        self.bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 96)
        self.hash = nn.Linear(96, 24)
        self.reg = nn.Linear(24, 8)
        self.fc2 = nn.Linear(96, 2)
        # self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)
        self.attention_weights = None

    def forward(self, x):
        x = F.leaky_relu(self.E2E(x) + self.E2E(x).transpose(3, 2))
        x = self.dropout(x)
        x = F.leaky_relu(self.E2N(x).transpose(3, 2) * 2)
        x = self.dropout(x)
        x = F.leaky_relu(self.N2G(x))
        x = self.dropout(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        self.attention_weights = x  # 保存注意力权重
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
    # key=lambda x: x[1]指定a排序的依据是每个元素的第二个元素（即数据量），reverse=True表示按照从大到小的顺序排序。
    sorted_site_data = sorted(site_data, key=lambda x: (x[1], x[0]), reverse=False)

    # 输出排序后的站点名称和对应的数据量
    for site, count in sorted_site_data:
        print(f"Site: {site}, Data Count: {count}")

    return sorted_site_data


def load_data(site):
    file = scio.loadmat('D:/2 Projects/Data for BNC/ABIDE1-work3/dataset/ASD_{}.mat'.format(site))
    X = file['net']
    Idx = [2, 3, 4, 5, 6, 7, 8, 9]  # 3:Age 4:Sex 5:Handedness 6:FIQ 7:VIQ 8:PIQ 9:EYE Status
    Y = file['phenotype'][:, Idx]

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

    # 用于存储读取的列表
    tri_list = []
    list_file = "D:/2 Projects/Data for BNC/ABIDE1-work3/Tri-list/{}_tri-list3.txt".format(site)
    # 打开文件以读取模式（'r' 表示读取）
    with open(list_file, 'r') as file:
        for line in file:
            # 解析文本行为Python列表
            current_list = eval(line.strip())  # 使用eval将文本行解析为列表
            tri_list.append(current_list)


    return roi, X, Y, Y_d, tri_list


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


def train_incremental(train_loader, X, Y, preserve, net, optimizer, ce_loss, reg_loss, sim_loss, triplet_loss):
    loss_sum = 0.

    train_hashcode = torch.empty(len(train_loader.dataset)*3, 24)
    train_hashcode_y = torch.empty(len(train_loader.dataset)*3)
    net.train()
    for step, batch_list in enumerate(train_loader):
        # sim = comput_similarity(batch_y.size(0), batch_y)
        ba = len(batch_list[0])
        # 使用 torch.cat 将列表中的张量连接成一个张量
        cl = torch.flatten(batch_list[0])  # combined_list

        batch_x = X[cl]
        batch_y = Y[cl]
        optimizer.zero_grad()
        IF, out_hash, out_reg, out_fc = net(batch_x)

        label = batch_y[:, 0].long()

        # in_pro = torch.mm(out_hash, out_hash.transpose(1, 0))
        J_CE = ce_loss(out_fc, label)
        J_MES = reg_loss(out_reg, batch_y)

        # 初始化 J_TRI
        J_TRI = 0.0
        # 计算每个三元组的损失并累加
        for i in range(0, 3*ba, 3):
            triplet_loss_value = triplet_loss(out_hash[i], out_hash[i + 1], out_hash[i + 2])
            J_TRI += triplet_loss_value
        # 求平均三元组损失
        J_TRI /= 3.0
        #J_TRI = triplet_loss(out_hash[0], out_hash[1], out_hash[2])
        loss_step = J_CE + J_MES + J_TRI  # 100 * ce_loss(out_fc, label) + 10 * sim_loss(in_pro, sim) + reg_loss(out_reg, batch_y) + triplet_loss(batch_x[0], batch_x[1], batch_x[2])
        hashcode = torch.div(torch.add(torch.sign(torch.sub(torch.sigmoid(out_hash), 0.5)), 1), 2)

        loss_step.backward(retain_graph=True)
        optimizer.step()

        loss_sum += loss_step.data.item()

        train_hashcode[step * train_loader.batch_size*3:step * train_loader.batch_size*3 + len(batch_x), :] = hashcode
        train_hashcode_y[step * train_loader.batch_size*3:step * train_loader.batch_size*3 + len(batch_x)] = batch_y[:, 0]

    loss = loss_sum / len(train_loader)

    # 如果有preserve数据，将preserve与当前输入合并
    # if preserve is not None:

    return loss, train_hashcode, train_hashcode_y





def infer(loader, net, ce_loss, reg_loss, sim_loss):
    loss_sum = 0.
    hc = torch.empty(len(loader.dataset), 24)
    hc_y = torch.empty(len(loader.dataset))
    net.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):
            sim = comput_similarity(batch_y.size(0), batch_y)

            IF, out_hash, out_reg, out_fc = net(batch_x)

            label = batch_y[:, 0].long()

            # in_pro = torch.mm(out_hash, out_hash.transpose(1, 0))
            loss_step = 100 * ce_loss(out_fc, label) + reg_loss(out_reg, batch_y)+triplet_loss(out_hash[0], out_hash[1], out_hash[2])
            hashcode = torch.div(torch.add(torch.sign(torch.sub(torch.sigmoid(out_hash), 0.5)), 1), 2)

            loss_sum += loss_step.data.item()


            hc[step * loader.batch_size:step * loader.batch_size + len(batch_x), :] = hashcode
            hc_y[step * loader.batch_size:step * loader.batch_size + len(batch_x)] = batch_y[:, 0]

        loss = loss_sum / len(loader)

    return loss, hc, hc_y

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

    BATCH_SIZE = 48
    EPOCH = 1000

    ce_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss(reduction='mean')
    sim_loss = nn.BCEWithLogitsLoss(reduction='mean')
    # 定义间隔（margin）的值
    margin = 0.5
    # 实例化三元组损失函数
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    sorted_site_data = sort_site()
    preserve = None



    for i, (test_site, _) in enumerate(sorted_site_data):
        print('test_site{}:{}'.format(i + 1, test_site))
        remaining_sites = [site for site, _ in sorted_site_data if site != test_site]

        valid_site = random.choice(remaining_sites)

        net = DHLNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

        # 构建训练集和测试集
        train_sites = [site for site in remaining_sites if site != valid_site]
        for j, site in enumerate(train_sites):

            ROI, X, Y, Y_d, Tri_list = load_data(site)
            train_dataset = Data.TensorDataset(torch.tensor(Tri_list))
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)  #会打乱顺序

            _, valid_X, valid_Y, valid_Y_d, _ = load_data(valid_site)
            valid_dataset = Data.TensorDataset(valid_X, valid_Y.to(torch.float32))
            valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

            _, test_X, test_Y, test_Y_d, _ = load_data(test_site)
            test_dataset = Data.TensorDataset(test_X, test_Y.to(torch.float32))
            test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            ax = []
            ay = []
            n_evaluation_epochs = 2
            plt.ion()

            # 定义终止条件
            early_stop_counter = 0  # 连续不下降的计数器
            best_loss = float('inf')  # 最佳验证损失
            max_early_stops = 10  # 连续不下降的最大次数

            for epoch in range(EPOCH):
                train_loss, train_hashcode, train_hashcode_y = train_incremental(train_loader, X, Y.to(torch.float32), preserve, net, optimizer, ce_loss, reg_loss, sim_loss, triplet_loss)
                print("site:{} train_loss:{}".format(site, train_loss))

                valid_loss, valid_hashcode, valid_hashcode_y = infer(valid_loader, net, ce_loss, reg_loss, sim_loss)
                valid_acc, sen, spe = comput_accuracy(valid_hashcode, train_hashcode, train_hashcode_y, valid_hashcode_y)
                print('valid_acc = ', valid_acc, 'valid_sen = ', sen, 'valid_spe = ', spe)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= max_early_stops:
                    print(f"Training terminated due to early stopping on site {test_site}.")
                    break

            _, test_hashcode, test_hashcode_y = infer(test_loader, net, ce_loss, reg_loss, sim_loss)
            test_acc, sen, spe = comput_accuracy(test_hashcode, train_hashcode, train_hashcode_y, test_hashcode_y)
            print('test_acc = ', test_acc, 'test_sen = ', sen, 'test_spe = ', spe)

                # if epoch % n_evaluation_epochs == 0:
                #     plt.figure(1)
                #     ax.append(epoch + 1)
                #     ay.append(np.mean(train_loss))
                #     plt.clf()
                #     plt.plot(ax, ay)
                #     plt.pause(0.01)
                #     plt.ioff()




        # 在这里，你可以选择是否保存模型参数以备下一轮训练使用
        # torch.save(net.state_dict(), "site_{}_model.pth".format(test_site))