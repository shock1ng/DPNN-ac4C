# -*- coding: utf-8 -*-
# @Time : 2024/1/9 17:28
# @Author : JohnnyYuan
# @File : PseKNC_seq0123_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAtt(nn.Module):
    def __init__(self, head=8):
        super(SelfAtt , self).__init__()
        self.head = head
        self.FC_q = nn.Linear(128, 128)
        self.FC_k = nn.Linear(128, 128)
        self.FC_v = nn.Linear(128, 128)
        self.drop = nn.Dropout(0.6)
        self.FC = nn.Linear(1024, 1024)

    def forward(self, x):
        batch_size, channels, width = x.size()   
    
        x = x.view(batch_size, channels, self.head, width // self.head)   
        x = torch.transpose(x,1,2)  
        q = self.FC_q(x)
        k = self.FC_k(x)
        v = self.FC_v(x)
        k = torch.transpose(k, -1, -2)   
        p_attn = nn.functional.softmax(torch.matmul(q, k) / math.sqrt( width // self.head ),dim=1)
    
        p_attn = self.drop(p_attn)
    
        att_out = torch.matmul(p_attn, v)
        att_out = att_out.view(batch_size, channels, width)
        att_out = self.FC(att_out)
    
        return att_out

# 构建位置编码器的类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.5, max_len=3000):

​        super(PositionalEncoding, self).__init__()

        self.drop = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)* -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe) 
    
    def forward(self , x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.drop(x)

class PseKNC_Seq_Classifier(nn.Module):
    def __init__(self):
        super(PseKNC_Seq_Classifier, self).__init__()
        self.drop = nn.Dropout(0.4)
        self.pool = nn.MaxPool1d(2)
        self.FC1 = nn.Linear(1024, 896)
        self.BiGRU = nn.GRU(input_size=66, hidden_size=512, dropout=0.5,
                                 num_layers=2, batch_first=True, bidirectional=True)
        self.self_att_Pse = SelfAtt_a_b(head = 8)    # 定义为8头

        self.CNNmodel = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Linear(9, 256) 
        )
    
        ###############  ###############
        self.embed = nn.Embedding(3000, 512)
        self.poistion_embed = PositionalEncoding(d_model=512, dropout=0.5)
        self.shapeChange = nn.Linear(201, 512)
        self.BiLSTM = nn.LSTM(input_size=512, hidden_size=512, dropout=0.5,
                              num_layers=2, batch_first=True, bidirectional=True)
        self.self_att = SelfAtt(head=8)
        self.FC2 = nn.Linear(1024, 128)
        ###############  ###############
    
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024, 256),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )


    def forward(self, seq, Pse):
        ############### 以下是PseKNC部分 ###############
    
        Pse = self.CNNmodel (Pse) 
        Pse = self.self_att_Pse(Pse)     
        Pse = self.drop(Pse)
    
        ############### 以上是PseKNC部分 ###############
    
        ###############  ###############
    
        onehot = seq.long()
        onehot_emb = self.embed(onehot) 
        onehot_emb = self.poistion_embed(onehot_emb)
        onehot_emb = self.drop(onehot_emb)
        lstm_out, _ = self.BiLSTM(onehot_emb) 
        # 变化都是为了更好地进入自注意力机制
        lstm_out = torch.transpose(lstm_out, 1, 2) 
        lstm_out = self.shapeChange(lstm_out) 
        lstm_out = torch.transpose(lstm_out, 1, 2)
        att_out = self.self_att(lstm_out)  
        # att_out = self.self_conv_att(lstm_out)       # 尝试卷积自注意力，带残差连接，开始训练就过拟合了，所以放弃
        att_out = torch.transpose(att_out, 1, 2)  
        one_out = torch.nn.functional.max_pool1d(att_out, att_out.size(2)).squeeze(2)  


        ############### ###############
    
        x = torch.cat([Pse, one_out], dim=1)
        y = self.classifier(self.drop(x))
    
        return y
