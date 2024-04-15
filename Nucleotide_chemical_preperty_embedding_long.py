# -*- coding: utf-8 -*-
# @Time : 2023/12/13 20:32
# @Author : JohnnyYuan
# @File : Nucleotide_chemical_preperty_embedding_long.py
# 我之前对NCP编码的理解是一串序列编码成[长度,3] ，有没有可能实际上是把这3个数字直接拼在list后面，而不是单独成一个维度呢？
import sys

import numpy as np
import pandas as pd
# 定义编码字典
encoding_dict = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'U': [0, 0, 1]}

def NCP(sequence):
    # 将RNA序列编码为列表
    encoded_list = []
    for char in sequence:
        encoded_list = encoded_list + encoding_dict[char]
    return encoded_list

# 打开文件
with open(r'/home/hd/JohnnyYuan/bio/RNA/Dataset/iRNA-ac4c/testset/iRNA-ac4c-testset.txt', 'r') as file:
    # 逐行读取文件内容并存储到列表
    lines = file.readlines()      # 生成列表，每个元素就是每一行的文本，但是最后都有换行符

NCP_Embedding = []
# 打印或处理列表中的每一行
for index, line in enumerate(lines):
    if index > 1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
        if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
            line = line.strip()   # 拿走每一行最后的换行符'\n'

            output = NCP(line)    # <class 'list'>
            NCP_Embedding.append(output)

NCP_Embedding_lie_df = pd.DataFrame(NCP_Embedding)  # train+val:[4412 rows x 603 columns]
print(NCP_Embedding_lie_df)
NCP_Embedding_lie_df.to_csv('NCP_Embedding_test_df.csv',index=False)
