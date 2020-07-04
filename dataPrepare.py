import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# 分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def oneSequence():

    # BERT 在预训练中引入了 [CLS] 和 [SEP] 标记句子的开头和结尾
    samples = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']  # 准备输入模型的语句
    for i in samples:
        print(i)
        print('')
    tokenized_text = [tokenizer.tokenize(i) for i in samples]
    print(tokenized_text)
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    input_ids = torch.LongTensor(input_ids)
    print(input_ids)
    return input_ids


def loadTxt():
    with open('./clothing_comment/negdata.txt', 'r', encoding='utf-8') as f:
        neg_data = f.read()
    with open('./clothing_comment/posdata.txt', 'r', encoding='utf-8') as f:
        pos_data = f.read()

    neg_datalist = neg_data.split('\n')     # 1500 ['', ]
    pos_datalist = pos_data.split('\n')
    print(len(neg_datalist), len(pos_datalist))
    print(neg_datalist[:5])
    print('')
    print(pos_datalist[:5])
    dataset = np.array(pos_datalist + neg_datalist)
    labels = np.array([1] * len(pos_datalist) + [0] * len(neg_datalist))
    # 共 3000 条数据  1 pos  0 neg
    # 随机打乱
    np.random.seed(10)
    # 从 0 ~ 3000 中随机抽 数字 3000 次  随机洗牌数据
    mix_index = np.random.choice(3000, size=3000)
    print(mix_index)
    print(len(mix_index))
    dataset = dataset[mix_index]
    labels = labels[mix_index]
    print(len(dataset), len(labels))
    # 训练集 2500 验证集 500
    TRAINSET_SIZE = 2500
    EVALSET_SIZE = 500

    train_samples = dataset[:TRAINSET_SIZE]  # 2500 条数据
    train_labels = labels[:TRAINSET_SIZE]
    eval_samples = dataset[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]  # 500 条数据
    eval_labels = labels[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]

    print(train_samples[0])
    print(train_labels[0])

    # dataLoader 构建 ----------------------------------------------------------------
    tokenized_text = [tokenizer.tokenize(i) for i in train_samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    print(input_ids[0])     # 这不是没有 [CLS] 和 [SEP] 了 ？？？
    input_labels = get_dummies(train_labels)  # 使用 get_dummies 函数转换标签
    print(input_labels[0])

    for j in range(len(input_ids)):
        # 将样本数据填充至长度为 512
        i = input_ids[j]
        if len(i) != 512:
            input_ids[j].extend([0] * (512 - len(i)))

    # 构建数据集和数据迭代器，设定 batch_size 大小为 4  2500/4 = 625
    train_set = TensorDataset(torch.LongTensor(input_ids), torch.FloatTensor(input_labels))
    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)

    print(len(train_loader))
    for f, l in train_loader:
        print(f)
        print(l)
        break

    # 验证集的 data-Loader
    tokenized_text = [tokenizer.tokenize(i) for i in eval_samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    input_labels = eval_labels

    for j in range(len(input_ids)):
        i = input_ids[j]
        if len(i) != 512:
            input_ids[j].extend([0] * (512 - len(i)))

    eval_set = TensorDataset(torch.LongTensor(input_ids), torch.FloatTensor(input_labels))
    eval_loader = DataLoader(dataset=eval_set, batch_size=1, shuffle=True)

    return train_loader, eval_loader


def get_dummies(l, size=2):
    """
    把标签 转换成 one - hot 形式 1表示成 [ 0, 1] 0 表示成 [1, 0]
    :param l: [1, 1, 1, 0, 0, 1, 0]
    :param size: 2位
    :return: [[0,1] or [1,0],....]
    """
    res = list()
    for i in l:
        tmp = [0] * size
        tmp[i] = 1
        res.append(tmp)
    return res


if __name__ == '__main__':

    # loadTxt()
    print(get_dummies([1, 1, 1, 0, 0, 1, 0]))

    trainloader, evalloader = loadTxt()


