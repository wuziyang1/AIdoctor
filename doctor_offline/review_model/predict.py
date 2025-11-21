import os
import torch
import torch.nn as nn

# 导入RNN模型结构
from RNN_MODEL import RNN
# 导入bert预训练模型编码函数
from bert_chinese_encode import get_bert_encode_for_single


# 预加载的模型参数路径
MODEL_PATH = './model/BERT_RNN-1.pth'

# 隐层节点数，输入层尺寸，类别数都和训练时相同即可
n_hidden = 128
input_size = 768
n_categories = 2

# 实例化RNN模型，并加载保存模型参数
rnn = RNN(input_size, n_hidden, n_categories)
rnn.load_state_dict(torch.load(MODEL_PATH))


def _test(line_tensor):
    """模型测试函数，它将用在模型预测函数中，用于调用RNN模型并返回结果。
    它的参数line_tensor代表输入文本的张量表示"""
    # 初始化隐层张量
    hidden = rnn.initHidden()#他返回的是一个三维数据
    # 与训练时相同，遍历输入文本的每一个字符
    for i in range(line_tensor.size()[1]):
        # 将其逐次输送给rnn模型
        output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)
    # 获得rnn模型最终的输出
    return output


def predict(input_line):
    """模型预测函数，输入参数input_line代表需要预测的文本"""
    # 不自动求解梯度
    with torch.no_grad():
        # 将input_line使用bert模型进行编码
        output = _test(get_bert_encode_for_single(input_line))
        # 从output中取出最大值对应的索引，比较的维度是1
        _, topi = output.topk(1, 1)
        # 返回结果数值
        return topi.item()


def batch_predict(input_path, output_path):
    """批量预测函数，以原始文本(待识别的命名实体组成的文件)输入路径
       和预测过滤后(去除掉非命名实体的文件)的输出路径为参数"""
    # 待识别的命名实体组成的文件是以疾病名称为csv文件名，
    # 文件中的每一行是该疾病对应的症状命名实体

    # 读取路径下的每一个csv文件名，装入csv列表之中
    csv_list = os.listdir(input_path)
    # 遍历每一个csv文件
    for csv in csv_list:
        # 以读的方式打开每一个csv文件
        with open(os.path.join(input_path, csv), "r") as fr:
            # 再以写的方式打开输出路径的同名csv文件
            with open(os.path.join(output_path, csv), "w") as fw:
                # 读取csv文件的每一行
                input_lines =fr.readline()
                for input_line in input_lines:
                    print(csv, input_line)
                    # 使用模型进行预测
                    res = predict(input_line)

                    if res: # 结果是1，说明审核成功，把文本写入到文件中
                        fw.write(input_line+'\n')
                    else:
                        pass

if __name__ == '__main__':
    input_path = '../data/structured/noreview'
    output_path = '../data/structured/review-wzy'
    batch_predict(input_path, output_path)


'''
中文分词，词性标注，命名实体识别都是序列标注问题
B表示开始
E表示结束
M表示中间
'''