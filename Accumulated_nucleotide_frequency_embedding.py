# -*- coding: utf-8 -*-
# @Time : 2023/12/12 14:59
# @Author : JohnnyYuan
# @File : Accumulated_nucleotide_frequency_embedding.py
import pandas as pd
def accumulated_nucleotide_frequency(sequence):
    """
    计算 DNA 或 RNA 序列的累积核苷酸频率

    参数:
    - sequence: DNA 或 RNA 序列，字符串形式（例如，'ATCGATCGATCG'）

    返回:
    - 一个包含每种核苷酸的累积频率的列表
    """
    # 初始化频率字典
    nucleotide_frequency = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
    # 遍历序列并计算频次
    for nucleotide in sequence:
        nucleotide_frequency[nucleotide] += 1
    encoded_list = [1/nucleotide_frequency[char] for char in sequence]
    return encoded_list

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized = [(val - min_val) / (max_val - min_val) for val in lst]
    return normalized

# 打开文件
with open(r'/home/hd/JohnnyYuan/bio/RNA/Dataset/iRNA-ac4c/testset/iRNA-ac4c-testset.txt', 'r') as file:   # 这里放置个人的文件路径
    # 逐行读取文件内容并存储到列表
    lines = file.readlines()      # 生成列表，每个元素就是每一行的文本，但是最后都有换行符

ANF_Embedding = []
# 打印或处理列表中的每一行
for index, line in enumerate(lines):
    if index > 1:   # 由于文本有前面几行的注释，这里选择 1 就是跳过前面两行，根据实际情况跳过不同行数
        if index % 2 != 0:   #  拿取奇数行的数据，因为偶数行是id
            line = line.strip()   # 拿走每一行最后的换行符'\n'
            output = accumulated_nucleotide_frequency(line)
            output = normalize_list(output)    # 自己选择要不要给特征做归一化
            ANF_Embedding.append(output)

ANF_Embedding_df = pd.DataFrame(ANF_Embedding)  # [4412 rows x 201 columns]    test: [1104 rows x 201 columns]
# 保存提取好的特征
ANF_Embedding_df.to_csv('ANF_Embedding_norm_test_df.csv',index=False)
