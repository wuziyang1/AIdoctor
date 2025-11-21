#todo:05-训练
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import BertTokenizer
from evaluate import evaluate

from bilstm_crf import NER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#todo:编码，打pad-这个函数也是数据预处理的--传进来的是一个batch的数据[[],[],[],[]]
def pad_batch_inputs(data, labels, tokenizer):

    # 函数需要返回一个按照内容长度从大到小排序过的，sentence 和 label, 还要返回 sentence 长度
    # 将批次数据的输入和标签值分开，并计算批次的输入长度
    data_inputs, data_length, data_labels = [], [], []
    for data_input, data_label in zip(data, labels):

        # todo:1-1对输入句子中的字进行编码 文本数值化
        data_input_encode = tokenizer.encode(data_input,#data_input是一句话
                                             return_tensors='pt',
                                             add_special_tokens=False)#是否有特殊的token
        data_input_encode = data_input_encode.to(device)#toeknizer返回出来的是[1,59]，把他里面的59个数取出来加入列表需要squeeze一下
        data_inputs.append(data_input_encode.squeeze())

        # todo:1-2去除多余空格只是为了计算计算句子长度
        data_input = ''.join(data_input.split())
        data_length.append(len(data_input))

        # todo:1-3将标签转换为张量
        data_labels.append(torch.tensor(data_label, device=device))

    # todo:2-1对一个批次的内容按照长度从大到小排序，符号表示降序
    sorted_index = np.argsort(-np.asarray(data_length))#这里是排序下标，把最长的挪到最前面，sorted_index是排好序的下标

    # todo:2-1根据长度的索引进行排序
    sorted_inputs, sorted_labels, sorted_length = [], [], []
    for index in sorted_index:
        sorted_inputs.append(data_inputs[index])
        sorted_labels.append(data_labels[index])
        sorted_length.append(data_length[index])

    sorted_length = torch.tensor(sorted_length, device=device)  # 新增

    #todo: 3-1对张量进行填充，使其变成长度一样的张量
    #print(sorted_inputs, )
    pad_inputs = pad_sequence(sorted_inputs)#这里没有指定多长，默认使用的是一个batch中最长的那个。
    # print(pad_inputs)

    return pad_inputs, sorted_labels, sorted_length


label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}


def train():
    # todo:-读取训练集
    train_data = load_from_disk('ner_data/bilstm_crf_data_aidoc')['train']

    # todo:-tokenizer分词器-刚才搞的那个词表就是在这儿用的
    tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')#BertTokenizer表明这里用的是bert的分词器
    '''
    使用自己训练的词表vocab.txt初始化了一个bert分词器，并不是直接使用预训练bert的词表
    tokenizer的作用是吧token转成id
    BertTokenizer作用有两个
        1.分词
        2.根据词表中每个字所在行的行号给字编号
        3.然后在到一句话中把字转成编号
    '''

    # 构建模型
    model = NER(vocab_size=tokenizer.vocab_size, label_num=len(label_to_index)).to(device)

    # 批次大小
    batch_size = 32
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    # 训练轮数
    num_epoch = 300

    # train history
    train_history_list = []

    # valid history
    valid_history_list = []

    def start_train(data_inputs, data_labels, tokenizer):
        # 对数据进行填充补齐
        pad_inputs, sorted_labels, sorted_length = pad_batch_inputs(data_inputs, data_labels, tokenizer)

        pad_inputs = pad_inputs.to(device)  # <--- 关键修改！

        # 计算损失
        loss = model(pad_inputs, sorted_labels, sorted_length)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计损失值
        nonlocal total_loss
        total_loss += loss.item()

    loss_list = []#损失函数曲线
    for epoch in range(num_epoch):
        # total_loss
        total_loss = 0.0

        train_data.map(start_train, input_columns=['data_inputs', 'data_labels'],
                       batched = True,
                       batch_size=batch_size,
                       fn_kwargs={'tokenizer': tokenizer},
                       desc='epoch: %d' % (epoch + 1))
        # input_columns会按照train.csv中的两列解包，把得到的结果赋值给data_inputs和data_labels

        print('epoch: %d loss: %.3f' % (epoch + 1, total_loss))
        loss_list.append(total_loss)

        if (epoch + 1) % 100 == 0:
            train_eval_result = evaluate(model, tokenizer, train_data)
            train_eval_result.append(total_loss)
            train_history_list.append(train_eval_result)

        if (epoch+1)%100==0:
            model.save_model('model/BiLSTM-CRF-%d.bin' % (epoch + 1))


    import matplotlib.pyplot as plt
    # 绘制损失函数曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epoch + 1), loss_list, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_curve.png')  # 保存图像（可选）
    plt.show()


if __name__ == '__main__':
    train()