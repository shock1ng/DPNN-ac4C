# -*- coding: utf-8 -*-
# @Time : 2023/12/21 13:26
# @Author : JohnnyYuan
# @File : iRNA_ac4c_2_NCP_ANF_train.py
import pandas as pd
import numpy as np
import csv
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold, StratifiedKFold
import os
import argparse
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix,matthews_corrcoef, roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.metrics import recall_score
import torch.nn.functional as F
from sklearn.metrics import fbeta_score

# from RNA.models.FGM_attack import FGM
from models import NCP_embed_ANF_Classifier, FGM    # 引入我的模型
from utils_iRNA_ac4c import Get_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="iRNA_ac4c_2_ncp_anf_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=128, metavar='N')
parser.add_argument('--val_batch_size', type=int, default=50, metavar='N')   # 一般别动
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)     
parser.add_argument('--fold',type=int, default=10, help='定义几折交叉验证')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=800)
parser.add_argument('--attack', type=bool, default=False)     # 增加攻击参数，选择True或者False
args = parser.parse_args()

### 定义焦损
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

### 定义二进制分类
def ce_loss(y_pred, y_train, alpha=1):
    # p = torch.sigmoid(y_pred)
    # p = torch.clamp(p, min=1e-9, max=0.99)
    loss = torch.sum(- alpha * torch.log(y_pred) * y_train
                     - torch.log(1 - y_pred) * (1 - y_train)) / len(y_train)
    return loss

### 定义二进制交叉熵的辅助矩阵
def get_target_weight(a):
    pos_weight = np.zeros([args.batch_size,2])
    for i in range(len(a)):
        if a[i][1] == 1:
            pos_weight[i][1] = 3
        if a[i][0] == 1:
            pos_weight[i][0] = 1
    pos_weight = torch.IntTensor(pos_weight)
    print(pos_weight)
    return pos_weight

def calculate_sen_spe(confusion_matrix):
    # Extract values from confusion matrix
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    # Calculate sensitivity (recall)
    sen = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    # Calculate specificity
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    return sen, spe

def Train(epoch, foldnum):
    train_loss = 0
    RNAnet.train()
    total_samples = 0
    correct_samples = 0
    for batch_idx, (_, _, _, _, anf, ncp, _,  label) in enumerate(train_loader):   # k2mer, k3mer, k4mer, k5mer, anf, ncp, label
        if args.cuda:
            ncp, anf, label = ncp.cuda(), anf.cuda(), label.cuda()

        net_optim.zero_grad()
        # k2mer = k2mer.view(args.batch_size, 1, -1)   # 增加通道数，因为第一个就是CNN
        # k3mer = k3mer.view(args.batch_size, 1, -1)
        # k4mer = k4mer.view(args.batch_size, 1, -1)
        # k5mer = k5mer.view(args.batch_size, 1, -1)
        # ncp = ncp.view(args.batch_size, 1 ,201, 3)
        anf = anf.view(args.batch_size, 1 ,201)

        y = RNAnet(ncp, anf)

        target = label.squeeze()    # torch.Size([64])

        # target_view = target.view(-1, 1)
        # target_one_hot = torch.nn.functional.one_hot(target, num_classes=2).cuda()
        # print(target_one_hot)
        # loss  = nn.CrossEntropyLoss()(y,target.long())
        loss = nn.BCELoss()(y.squeeze(),target.float())
        # loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([3]).cuda())(y, target_one_hot)  # 按照AVP / NonAVP = 3:1, 这里的pos_weight要记得放在cuda上面
        # loss = FocalLoss(alpha=3, gamma=2)(y,target.long())
        # loss = ce_loss(y, target,alpha=3)
        loss.backward()

        #### 尝试加入attcak ####
        if args.attack == True :
            FGMModel = FGM(RNAnet, 'NCP_embed_ANF_Classifier.embed')
            FGMModel.attack()
            y_attacked = RNAnet(ncp, anf)
            loss_attacked = nn.CrossEntropyLoss()(y_attacked, target.long())
            loss_attacked.backward()
            FGMModel.restore()
        #### 尝试加入attcak ####

        net_optim.step()
        train_loss += loss
        # _, predicted = torch.max(y, 1)   # 输出是2的时候用，这里的pred是(batch,)  不是[batch,2]，不是二维的了
        predicted = torch.tensor([0 if p[0] < 0.5 else 1 for p in y]).cuda()
        total_samples += target.size(0)
        correct_samples += (predicted == target).sum().item()
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.2f}%    \tTrain Fold:[{}/{}]'.format(
                epoch, args.epochs , batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval,
                       (correct_samples / total_samples) * 100,
                        foldnum, args.fold
            ))
            train_loss = 0


def Val():
    RNAnet.eval()
    label_pre = []
    label_true = []

    label_pred_for_auc = []

    with torch.no_grad():
        for batch_idx, (_, _, _, _, anf, ncp, _,  label) in enumerate(train_loader):
            if args.cuda:
                 ncp, anf, label = ncp.cuda(), anf.cuda(), label.cuda()


            net_optim.zero_grad()
            # k2mer = k2mer.view(args.batch_size, 1, -1)  # 增加通道数，因为第一个就是CNN
            # k3mer = k3mer.view(args.batch_size, 1, -1)
            # k4mer = k4mer.view(args.batch_size, 1, -1)
            # k5mer = k5mer.view(args.batch_size, 1, -1)
            # ncp = ncp.view(args.batch_size, 1, 201, 3)
            anf = anf.view(args.batch_size, 1, 201)

            y = RNAnet(ncp, anf)
            # y = y.squeeze()

            # output = torch.argmax(y, dim=1)   # 在列方向上找到最大值的索引
            output = torch.tensor([0 if p[0] < 0.5 else 1 for p in y])
            label_true.extend(label.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
            label_pred_for_auc.extend(y.cpu().data.numpy())
            # print(label_true)
            # print(label_pre)

            # 这里的函数是来自sklearn的，用于计算召回值，f1值，混淆矩阵
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        # 计算MCC
        mcc = matthews_corrcoef(label_true, label_pre)
        # 计算AUC
        auc = roc_auc_score(label_true, label_pre)  # 假设正例的类别是1
        # 计算Sensitivity和Specificity
        sen , spe = calculate_sen_spe(CM_test)
        print("===================================================================== 以下是测试数据 ===========================================================================")
        print("Recall:", accuracy_recall)
        print("ACC:", accuracy_f1)
        print("MCC:", mcc)
        print("AUC:", auc)
        print("Sensitivity:", sen)
        print("Specificity:", spe)
        print(CM_test)
        print("########" * 20)


        ##### 开始记录auc信息 #####
        tg_true = []
        tg_pred = []
        if auc > 0.82:
            print("保存AUC的list咯！")
            for idx in range(len(label_true)):
                tg_true.append(label_true[idx][0])
                tg_pred.append((label_pred_for_auc[idx]))
            with open("auc_true.csv", "a", encoding="utf-8", newline="") as f:   # a 代表追加
                csv_writer = csv.writer(f)
                csv_writer.writerow(tg_true)

            with open("auc_pred.csv", "a", encoding="utf-8", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(tg_pred)

            print("-------------------------------------保存auc数据-------------------------------------")

        ########################


    return accuracy_f1, accuracy_recall, label_pre, label_true, CM_test, mcc, auc , sen, spe

data_save = "iRNA_ac4c_attack_dropMore05_val.txt"

## 加载部分保存好的数据，并且生成标签
pd2mer = pd.read_csv('../RNA/Dataset/iRNA-ac4c/trainset/list2mer_4412_df.csv')
# label
label_pos = np.ones((2206, 1))
label_neg = np.zeros((2206, 1))
label = np.append(label_pos, label_neg)
label = pd.DataFrame(label)

# 使用StratifiedKFold进行分层抽样，4折交叉验证
sum_cm = np.zeros((2, 2))
sum_sen = 0
sum_spe = 0
sum_mcc = 0
sum_auc = 0
kfold = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)   # 模型里面太多不同的随机数对结果的影响太大了
# kfold = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)   # 模型里面太多不同的随机数对结果的影响太大了
# stratified_kfold = KFold(n_splits=num_folds, shuffle=True)
for fold_idx, (train_index, val_index) in enumerate(kfold.split(pd2mer.values , label.values)):
    print('=' * 20)
    print('fold：',fold_idx+1)
    print("train_idx长度：",len(train_index))
    print("val_idx长度：",len(val_index))

    train_loader, val_loader = Get_data(train_index, val_index, args)

    RNAnet = NCP_embed_ANF_Classifier(args).cuda()   # 声明网络
    lr = args.lr
    net_optimizer = getattr(optim, args.optim)(RNAnet.parameters(), lr=lr)
    net_optim = optim.Adam(RNAnet.parameters(), lr=lr)
    f1 = 0
    recall = 0
    recall_list = []
    f1_list = []
    cm_list = []
    mcc_list = []
    auc_list = []
    sen_list = []
    spe_list = []

    for epoch in range(1, args.epochs + 1):
        ############################################################ 开始训练 ############################################################
        Train(epoch, fold_idx+1)
        accuracy_f1, accuracy_recall, pre_label, true_label, cm, mcc, auc ,sen, spe= Val()
        recall_list.append(accuracy_recall)
        f1_list.append(accuracy_f1)
        cm_list.append(cm)
        mcc_list.append(mcc)
        auc_list.append(auc)
        sen_list.append(sen)
        spe_list.append(spe)
        if epoch % 15 == 0:
            lr /= 10    # 学习率衰减
            for param_group in net_optimizer.param_groups:
                param_group['lr'] = lr

        if (accuracy_f1 > f1 and accuracy_recall > recall):
            name_1 = 'RNAnet' + str(fold_idx) + '.pkl'
            torch.save(RNAnet.state_dict(), name_1)
            recall = accuracy_recall
            f1 = accuracy_f1
    max_recall = max(recall_list)
    max_f1 = f1_list[recall_list.index(max_recall)]  # 通过在recall列表里检索下标来输出对应的f1数值
    cm = cm_list[recall_list.index(max_recall)]
    mcc = mcc_list[recall_list.index(max_recall)]
    auc = auc_list[recall_list.index(max_recall)]
    sen = sen_list[recall_list.index(max_recall)]
    spe = spe_list[recall_list.index(max_recall)]
    sum_cm += cm
    sum_mcc += mcc
    sum_auc += auc
    sum_sen += sen
    sum_spe += spe
    print("成功统计该折数据")
    with open(data_save, 'a') as f:
        f.write("\n" +"第" + str(fold_idx + 1) + "折数据：" + "\n" + 'Max_recall:\t'+str(max_recall) + '\n' +'对应的f1:\t'+ str(max_f1) + '\n' +
                '对应的mcc:' + str(mcc) + '\n' +'对应的auc:' + str(auc)+ '\n' +'对应的sen:' + str(sen)+  '\n' +'对应的spe:' + str(spe)+ '\n' + str(cm) + '\n')
        print("输出结果已保存")

with open(data_save, 'a') as f:
    f.write(f'\n{args.fold}个最佳混淆矩阵之和是：\n' + str(sum_cm) + '\n' +
            '平均AUC: \t' + str(sum_auc/args.fold) + '\n' +
            '平均MCC：\t' + str(sum_mcc/args.fold) + '\n' +
            '平均SEN：\t' + str(sum_sen/args.fold) + '\n' +
            '平均SPE：\t' + str(sum_spe/args.fold))
    print("最终混淆矩阵：\n", sum_cm)
    print('\n平均AUC: \t' + str(sum_auc/args.fold) + '\n' +
            '平均MCC：\t' + str(sum_mcc/args.fold) + '\n' +
            '平均SEN：\t' + str(sum_sen/args.fold) + '\n' +
            '平均SPE：\t' + str(sum_spe/args.fold))
    print("最终结果已保存")

def write_lists_to_csv(list1, list2, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for l1, l2 in zip(list1, list2):
            writer.writerow([l1, l2])

