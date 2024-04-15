# -*- coding: utf-8 -*-
# @Time : 2023/12/11 18:28
# @Author : JohnnyYuan
# @File : K-mer序列编码.py
import numpy as np
import torch
import pandas as pd
def kmer_composition(sequence, k):
    """
    计算 DNA 序列的 K-mer nucleotide composition
    参数:
    - sequence: 序列，字符串形式（例如，'ATCGATCGATCG'）
    - k: K-mer 的长度
    返回:
    - 一个字典，包含每个 K-mer 的频率
    """
    composition = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]    # 对序列进行切片
        composition[kmer] = composition.get(kmer, 0) + 1   # 在composition字典里查找 有没有 kmer这个字段，没有的话返回0 + 1，有的话返回对应数值 + 1

    seqEmbed = []
    for j in range(len(sequence) - k + 1):
        kkmer = sequence[j:j + k]  # 对序列进行切片
        num = composition.get(kkmer)
        seqEmbed.append(num/(len(sequence) - k + 1))

    return composition ,seqEmbed

# 打开文件
with open(r'/home/hd/JohnnyYuan/bio/RNA/Dataset/iRNA-ac4c/trainset/iRNA-ac4c-trainset-4410.txt', 'r') as file:
    # 逐行读取文件内容并存储到列表
    lines = file.readlines()      # 生成列表，每个元素就是每一行的文本，但是最后都有换行符

list2mer = []
list3mer = []
list4mer = []
list5mer = []

# 打印或处理列表中的每一行
for index, line in enumerate(lines):
    if index > 1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
        if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
            line = line.strip()   # 拿走每一行最后的换行符'\n'
            for kmer in range(2, 6):  # K-mer的范围是2,3,4,5
                _, seqEmbedding = kmer_composition(line, kmer)

                if kmer == 2:
                    list2mer.append(seqEmbedding)
                if kmer == 3:
                    list3mer.append(seqEmbedding)
                if kmer == 4:
                    list4mer.append(seqEmbedding)
                if kmer == 5:
                    list5mer.append(seqEmbedding)


list2mer_df = pd.DataFrame(list2mer)   # [4412 rows x 200 columns]
list3mer_df = pd.DataFrame(list3mer)   # [4412 rows x 199 columns]
list4mer_df = pd.DataFrame(list4mer)   # [4412 rows x 198 columns]
list5mer_df = pd.DataFrame(list5mer)   # [4412 rows x 197 columns]
# 保存df方便调用
list2mer_df.to_csv('list2mer_4410_df.csv',index=False)
list3mer_df.to_csv('list3mer_4410_df.csv',index=False)
list4mer_df.to_csv('list4mer_4410_df.csv',index=False)
list5mer_df.to_csv('list5mer_4410_df.csv',index=False)

