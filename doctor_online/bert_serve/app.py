#todo:4-部署
# 这里只是部署句子相关模型并没有涉及前面的eno4j和redis
from flask import Flask
from flask import request
from flask import send_from_directory
app = Flask(__name__)

import torch
# 导入中文预训练模型编码函数
from bert_chinese_encode import get_bert_encode
# 导入微调网络
from finetuning_net import Net

# 导入训练好的模型
MODEL_PATH = "./model/BERT_net.pth"
# 定义实例化模型参数
embedding_size = 768
char_size = 20
dropout = 0.2

# 初始化网络
net = Net(embedding_size=embedding_size, char_size=char_size, dropout=dropout)
# 加载模型参数
net.load_state_dict(torch.load(MODEL_PATH))

# 使用评估模式
net.eval()
# 定义接口，这里是通过网络请求到这里的，不是函数调用
@app.route('/v1/recognition/', methods = ["POST"])
def recognition():
    text1 = request.form['text1']
    text2 = request.form['text2']
    print("recognition:", text1, text2)
    # 对文本进行编码
    inputs = get_bert_encode(text1, text2, mark=102, max_len=10)

    # 把编码输入到模型
    outputs = net(inputs)

    # 获得预测结果
    _, predicted = torch.max(outputs, 1)

    # 返回结果
    return str(predicted.item())



# 登录页路由
@app.route('/login', methods=["GET"])
def login_page():
    return send_from_directory('.', 'login.html')


@app.route('/login', methods=["POST"])
def login_submit():
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    # 演示用：不做真实校验
    return f"欢迎，{username}！(演示环境，不做认证)"













