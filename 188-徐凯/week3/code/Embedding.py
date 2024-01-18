import torch
import torch.nn as nn

"""
embedding层的处理
"""

num_embeddings = 7  # 通常对于 nlp 任务，此参数为字符集总数
embedding_dim = 5  # 每个字符向量化后的向量维度

embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
print("随机初始化权重")
print(embedding_layer.weight)
print("#" * 20)

# 构造字符表
vocab = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
}


def str_to_sequence(string, vocab):
    return [vocab[s] for s in string]


string1 = "abcde"
string2 = "ddccb"
string3 = "fedab"

sequence1 = str_to_sequence(string1, vocab)
sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)

print(sequence1)
print(sequence2)
print(sequence3)

x = torch.LongTensor([sequence1, sequence2, sequence3])
embedding_out = embedding_layer(x)
print(embedding_out)  # 3*5*5 = batch_size*length*emd_dim


# padding
def padding(max_length, sequence):
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        return sequence + [0] * (max_length - len(sequence))


sequence1_padded = padding(7, sequence1)
sequence2_padded = padding(7, sequence2)
sequence3_padded = padding(7, sequence3)

print(sequence1_padded)
print(sequence2_padded)
print(sequence3_padded)
