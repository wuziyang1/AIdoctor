import torch
import torch.nn as nn


class RNN(nn.Module):#整个类就只实现了一个时间部的功能。两个这样的类堆叠起来就是两个实践部
    def __init__(self, input_size, hidden_size, output_size):
        """

        :param input_size: 输入张量最后一个维度的大小，最后一个维度其实就是特征维度
        :param hidden_size: 隐藏层张量最后一个维度的大小
        :param output_size: 输出层张量最后一个维度的大小
        """
        super(RNN, self).__init__()

        # 保存hidden_size
        self.hidden_size = hidden_size

        # 构建第一个线性层
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)

        # tanh层
        self.tanh = nn.Tanh()

        # 第二个线性层
        self.i2o = nn.Linear(hidden_size, output_size)

        # 定义最后的softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden1):
        """

        :param input1: x(t)本时间步的输入
        :param hidden1: h(t-1)上一个时间步的输出
        :return:
        """

        # 拼接x(t) h(t-1)
        combined = torch.cat((input1, hidden1), dim=1)

        # 进入第一个Linear
        hidden = self.i2h(combined)

        # tanh层
        hidden = self.tanh(hidden)

        # 进入第二个Linear层
        output = self.i2o(hidden)#第二个线性层是用来分类的不是原本RNN内部结构

        # 进入softmax
        output = self.softmax(output)

        return output, hidden


    def initHidden(self):
        # 初始化 [1, hidden_size] 全为0的张量
        return torch.zeros(1, self.hidden_size)


if __name__ == '__main__':
    # 定义参数
    input_size = 768 # bert模型输出的维度
    hidden_size = 128 # 自定义的
    n_categories = 2 # 类别数量

    input = torch.rand(1, input_size) # 随机生成符合形状的张量
    print(input.size())
    hidden = torch.rand(1, hidden_size)

    rnn = RNN(input_size, hidden_size, n_categories) # 实例化模型
    outputs, hidden = rnn(input, hidden)
    print("outputs:", outputs)
    print("hidden:", hidden)
