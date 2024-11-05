# -*- coding: utf-8 -*-
# @Time : 2024/1/9 17:31
# @Author : JohnnyYuan
# @File : utils_PseKNC_seq.py
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np

## 加载保存好的数据，并且生成标签
# try_df = pd.read_csv('../RNA/Dataset/iRNA-ac4c/trainset/onehot_Embedding_train_lie_df.csv')   # onehot
Pse_df = pd.read_csv('../RNA/Dataset/iRNA-ac4c/trainset/PseKNC_4412_66forNN.csv')             # PseKNC
seq_df = pd.read_csv('../RNA/Dataset/iRNA-ac4c/trainset/seq_Embedding_train_4412_0123_df.csv')  # 0123

# label
label_pos = np.ones((2206, 1))
label_neg = np.zeros((2206, 1))
label = np.append(label_pos, label_neg)
label = pd.DataFrame(label)

class SubDataSet(Data.Dataset):
    # pse, seq
    def __init__(self, data1, data2,  label):
        # 因为数据来源是DF,不加values就不行
        self.Data1 = data1.values
        self.Data2 = data2.values
        self.Label = label.values

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, idx):   # 以idx来查询
        data_1 = torch.Tensor(self.Data1[idx])
        data_2 = torch.Tensor(self.Data2[idx])
        label = torch.Tensor(self.Label[idx])
        return data_1, data_2,  label

def Get_data(train_index, val_index, args):
    train_pse , val_pse = Pse_df.iloc[train_index], Pse_df.iloc[val_index]
    train_seq, val_seq = seq_df.iloc[train_index], seq_df.iloc[val_index]
    label_train, label_val = label.iloc[train_index], label.iloc[val_index]

    trainDataSet = SubDataSet(train_pse, train_seq, label_train)
    valDataSet = SubDataSet(val_pse, val_seq, label_val)
    train_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=args.batch_size, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(valDataSet, batch_size=args.val_batch_size, drop_last=True, shuffle=True)

    return train_loader , test_loader
