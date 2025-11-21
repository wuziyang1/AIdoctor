# 服务框架使用Flask
# 导入必备的工具
from flask import Flask
from flask import request
app = Flask(__name__)

# 导入发送http请求的requests工具
import requests

# 导入操作redis数据库的工具
import redis

# 导入加载json文件的工具
import json

# 导入已写好的闲聊机器人Unit API调用文件
from unit import unit_chat

# 导入操作neo4j数据库的工具
from neo4j import GraphDatabase

# 从配置文件中导入需要的配置
# NEO4J的连接配置
from config import NEO4J_CONFIG
# REDIS的连接配置
from config import REDIS_CONFIG
# 句子相关模型服务的请求地址
from config import model_serve_url
# 句子相关模型服务的超时时间
from config import TIMEOUT
# 规则对话模版的加载路径
from config import reply_path
# 用户对话信息保存的过期时间
from config import ex_time

# 建立REDIS连接池
pool = redis.ConnectionPool(**REDIS_CONFIG)

# 初始化NEO4J驱动对象
_driver = GraphDatabase.driver(**NEO4J_CONFIG)


# todo:1-根据用户输入的症状进行疾病名称的查询
def query_neo4j(text):
    """
    :param text: 症状名称，输入进来的是一句话
    :return: 对应的疾病名称
    """
    with _driver.session() as session:
        #匹配句子中存在的所有症状的节点
        #保存这些节点并逐一通过关系dis_to_sym进行对应症状的查找，返回找到的疾病名称列表
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH" \
                 " a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" % text
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))

    return result


# 实现处理不同分支函数
class Handler(object):
    def __init__(self, uid, text, r, reply): # 类内部的函数，第一个参数是self
        self.uid = uid
        self.r = r#redis数据库连接对象
        self.text = text
        self.reply = reply#reply.json对话模板

    #todo:1-用户不是第一次来
    def non_first_sentence(self, previous):
        #previous是用户输入本文的上一句文本
        try:
            print("准备请求句子相关模型服务!")
            data = {'text1':previous, 'text2':self.text}

            #todo:2-向句子主题相关模型发送请求
            result = requests.post(model_serve_url, data=data, timeout=TIMEOUT)
            print("句子相关模型服务请求成功, 返回结果为:", result.text)

            #todo:3-如果两句话不相关，把当前发的这句话传到闲聊机器人中
            if not result.text: return unit_chat(self.text)
            # 也可以，如果前后两句没有相关性，把第二句的text放到neo4j中查询
            # 也可以返回查询的结果
            # print("non_first_sentence, unit_chat")
            # return unit_chat(self.text)
        except Exception as e:
            print("模型异常", e)
            return unit_chat(self.text)

        #todo:4-如果相关，开始调用neo4j查询当前句子中的疾病症状
        s = query_neo4j(self.text)
        print(s)
        #todo:4-1如果这句话中没有相关症状
        if not s: return unit_chat(self.text)

        #todo:4-2如果当前这句话查到了结果，我就拿当前结果和上一次查询的结果求并集然后在减去上一次查询的结果，这样得到的就是这一轮查找新增加的
        old_disease = self.r.hget(str(self.uid), "previous_d")

        #todo:4-3如果成功把第一次存的结果拿到
        if old_disease:
            new_disease = list(set(s)|set(old_disease))
            res = list(set(s)-set(old_disease)) # 在新的集合中存在且不在之前集合中存在的元素
        #todo:4-4如果发现之前的结果为空，就不做并集和减法了
        else:
            res = list(set(s))
            new_disease = res

        #todo:然后把当前查询到的存起来作为下一次查询的‘上一次’
        self.r.hset(str(self.uid), "previous_d", str(new_disease))

        self.r.expire(str(self.uid), ex_time)#设置超时10h

        if not res:
            return self.reply['4']
        else:
            res = '，'.join(res)
            return self.reply['2'] % res

    #todo:2-用户第一次来
    def first_sentence(self):
        # 处理用户第一次输入
        s = query_neo4j(self.text)

        #todo:如果没有查到任何与疾病有关的信息
        if not s:
            return unit_chat(self.text)

        #todo:将s存储下来
        self.r.hset(str(self.uid), 'previous_d', str(s))
        self.r.expire(str(self.uid), ex_time) #超时时间

        res = "，".join(s)#肠炎,胃炎,阑尾炎

        return self.reply['2'] % res


# todo:实现主要逻辑服务，是通过发送post网络请求
@app.route("/v1/main_serve/", methods=["POST"])#/v1/main_serve/是接口名称
def main_serve():
    # 接收werobot发送的请求
    uid = request.form['uid']
    text = request.form['text']
    # todo:1-从redis中获取一个活跃的连接
    r = redis.StrictRedis(connection_pool=pool)

    # todo:2-根据用户的uid获取是否存在上一句话
    previous = r.hget(str(uid), 'previous')#uid和previous一起作为key来查找value
    print("main_serve previous:", previous)

    # todo:3-设置当前的输入作为上一句
    r.hset(str(uid), 'previous', text)

    # 获取模板
    reply = json.load(open(reply_path, 'r'))

    # todo:构造handler，根据是否是第一次请求实现逻辑
    handler = Handler(uid, text, r, reply)

    # todo:根据previous是否存在，判断是否是第一句
    if previous:
        return handler.non_first_sentence(previous)
    else:
        return handler.first_sentence()


# # 测试query_neo4j函数
# if __name__ == '__main__':
#     text = "我最近腹痛!"
#     print(query_neo4j(text))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

'''
我们之前做的命名实体识别和命名实体审核的时候的那一系列操作在doctor_online中都用不到
实体识别和审核都是为了处理我们爬虫爬进来的数据的
'''