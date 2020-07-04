import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel
from torch import optim
from torch.autograd import Variable
import time
from BERTChinese.dataPrepare import loadTxt

from transformers import AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

modelPath = 'bert-chinese'
train_loader, eval_loader = loadTxt()


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        # self.model = AutoModelWithLMHead.from_pretrained(modelPath)
        # self.model = BertModel.from_pretrained(model_name, cache_dir="./")
        self.model = BertModel.from_pretrained(modelPath)       # 模型的上层文件夹的路径
        print("预训练模型 加载完毕 bert chinese")

        self.model.to(device)

        # 随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。
        self.dropout = nn.Dropout(0.1)

        self.l1 = nn.Linear(768, 2)

    def forward(self, x, attention_mask=None):
        # x [4, 512] attention_mask [4, 512]
        # BERT-base 词向量为 768 为 Max Sequence 为 512 /  BERT-large 词向量为 1024维
        outputs = self.model(x, attention_mask=attention_mask)          # ([4, 512, 768], [4, 768])
        # 他这个完全不太对啊 这个 在数据上 没有加 CLS 和 [SEP] 而且 最后的输出 怎么是 2个
        x = outputs[1]  # 取池化后的结果 batch * 768

        x = x.view(-1, 768)
        x = self.dropout(x)     # batch * 768       以0.1的概率 置零
        x = self.l1(x)          #
        return x


def predict(logits):
    res = torch.argmax(logits, 1)
    return res


def train(train_loader):
    training_stats = []  # 存储训练信息


    clsNet = fn_cls()
    clsNet.to(device)
    clsNet.train()

    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    optimizer = optim.Adam(clsNet.parameters(), lr=1e-5)

    pre = time.time()

    accumulation_steps = 8
    epoch = 3

    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(device), Variable(target.view(-1, 2)).to(device)

            # Attention Mask
            mask = []
            for sample in data:
                mask.append([1 if i != 0 else 0 for i in sample])
            mask = torch.Tensor(mask).to(device)

            output = clsNet(data, attention_mask=mask)

            # 将较大值 变 1  较小值 变 0
            pred = predict(output)

            # ???? 这特么 什么操作啊 把 sigmoid 放在外面 这 网络里最后一层不就完事了
            loss = criterion(sigmoid(output).view(-1, 2), target)

            # 梯度积累
            loss = loss / accumulation_steps
            loss.backward()

            # 这操作到 还是挺亮眼的
            if ((batch_idx + 1) % accumulation_steps) == 0:
                # 每 8 次更新一下网络中的参数
                optimizer.step()
                optimizer.zero_grad()

            if ((batch_idx + 1) % accumulation_steps) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                        i + 1, batch_idx, len(train_loader), 100. *
                        batch_idx / len(train_loader), loss.item()))

                # Record all statistics from this epoch.
                training_stats.append({'epoch': i + 1,
                                       'Batch Percentage': (100. *batch_idx / len(train_loader)),
                                       'Training Loss': loss.item()})

            if batch_idx == len(train_loader) - 1:
                # 在每个 Epoch 的最后输出一下结果
                print('labels:', target)
                print('pred:', pred)

    print('训练时间：', time.time() - pre)

    torch.save(clsNet, 'clsNet.pt')
    print("模型保存完毕")

    return training_stats


def eval(eval_loader):
    model = torch.load('clsNet.pt')
    model.eval()

    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(eval_loader):
        data = data.to(device)
        target = target.long().to(device)

        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = model(data, attention_mask=mask)
        pred = predict(output)

        correct += (pred == target).sum().item()
        total += len(data)

    # 准确率应该达到百分之 90 以上
    print('正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}%'.format(correct, total, 100. * correct / total))


# 模型 非常 focus on 句子的开头内容
def mineSentence():
    test_samples = ['菜太难吃了，再也不会来了，但是餐馆的服务不错']

    model = torch.load('clsNet.pt')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    tokenized_text = [tokenizer.tokenize(i) for i in test_samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    input_ids = torch.LongTensor(input_ids).cuda()

    mask = torch.ones_like(input_ids).to(device)

    output = model(input_ids, attention_mask=mask)
    pred = predict(output)
    print("输入的语句：", test_samples)
    ans = pred.cpu().numpy()
    print(ans)
    if ans[0] == 1:
        print('判断为 [好评]')
    else:
        print('判断为 [差评]')
    # print('Tips： 1 是 好评， 0 是 差评')


def showTraning_stats(training_stats):
    import pandas as pd

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    print(df_stats)


if __name__ == '__main__':
    # training_stats = train(train_loader)
    # print(time.ctime())
    # showTraning_stats(training_stats)
    # print(time.ctime())
    # eval(eval_loader)       # 正确分类的样本数：473，样本总数：500，准确率：94.60%
    print(time.ctime())
    mineSentence()
    print(time.ctime())

