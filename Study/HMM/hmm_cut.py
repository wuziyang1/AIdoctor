#todo: 使用hmm模型进行分词

import numpy as np
import os
import pickle


class HMM(object):
    def __init__(self):
        self.pi = None
        self.A = None # 转移矩阵
        self.B = None # 发射概率矩阵，观测概率矩阵
        self.model_file = 'hmm_model.pkl'
        self.state_list = ['B', 'M', 'E', 'S'] # 表示的是每个在词语中的位置
        self.load_param = False

    def try_load_model(self, trained):
        if trained:
            with open(self.model_file, 'rb') as f:
                self.A = pickle.load(f)
                self.B = pickle.load(f)
                self.pi = pickle.load(f)
                self.load_param = True
        else:
            self.A = {}
            self.B = {}
            self.pi = {}
            self.load_param = False

    #todo:这是在构建隐马模型三要素PI A B的，没有涉及计算
    def train(self, path):
        #todo:1.加载参数
        self.try_load_model(False)

        #todo:2-把PI A B初始化成0
        def init_parameters(): #用字典模拟矩阵
            for state in self.state_list:
                self.A[state] = { s: 0.0 for s in self.state_list}
                self.B[state] = {}
                self.pi[state] = 0.0
                #经过初始化后
                '''
                    # A = {'B':{'B':0, 'M':0, 'E':0, 'S':0},转移矩阵
                    #      'M':{'B':0, 'M':0, 'E':0, 'S':0},
                    #      'E':{'B':0, 'M':0, 'E':0, 'S':0},
                    #      'S':{'B':0, 'M':0, 'E':0, 'S':0}}
                    # B = {'B':{}, 'M': {}, 'E': {}, 'S': {}}观测矩阵
                    # pi =  {'B':0, 'M': 0, 'E': 0, 'S': 0}初始值
                    
                    对于# B = {'B':{}, 'M': {}, 'E': {}, 'S': {}}观测矩阵
                        'B':{'迈':1,}, 'M': {'向':1}, 'E': {}, 'S': {}
                        这里的BMSE分别表示四个盒子
                        '迈'这个字表示盒子中不同颜色的球
                '''

        # todo:3-这是把文字转成BMSE这种序列
        def generate_state(text):
            state_seq = []
            if len(text) == 1:#传进来一个单字
                state_seq.append('S')
            else:
                state_seq += ['B'] + ['M']*(len(text)-2) + ['E']

            return state_seq

        init_parameters()

        #todo:主体业务逻辑
        with(open(path, encoding='utf8')) as f:#打开训练路径
            for line in f:
                # 更 高 地 举起 邓小平理论 的 伟大 旗帜”这句话及其对应状态序列“SSSBEBMMMESBEBE
                line = line.strip()
                if not line:
                    continue

                # 把一行的每个字 生成列表
                word_list = [i  for i in line if i != ' ']# word_list [更 高 地 举 起 邓 小 平 理 论 的 伟 大 旗 帜]

                # 文本按照空格进行分割
                line_sequences = line.split()
                line_states = []
                for seq in line_sequences:#取出手动分好词后的每一个单词
                    # 取出每个单词，转成对应的状态
                    line_states.extend(generate_state(seq))# line_states = SSSBEBMMMESBEBE

                # todo:根据一行中B M S E计算A B pi初始值
                for idx, state in enumerate(line_states):#state是BMSE中的一种
                    if idx == 0:
                        self.pi[state] += 1#todo:1.初始概率PI
                    else:
                        self.A[line_states[idx-1]][state] += 1#todo:idx-1是上一个状态，
                    self.B[line_states[idx]][word_list[idx]] = \
                        self.B[line_states[idx]].get(word_list[idx], 0) + 1.0#先确定是哪个盒子再确定是哪种颜色的球，然后+1

            # 计算pi初始概率
            self.pi = {k: np.log(v / np.sum(list(self.pi.values()))) if v != 0 else -3.14e+100 for k, v in self.pi.items()}

            # 计算转移概率
            self.A = { k:{ k1 : np.log(v1/np.sum(list(v.values()))) if v1 != 0 else -3.14e+100 for k1, v1 in v.items()} for k, v in self.A.items()}

            # 计算发射概率
            self.B = { k: {k1: np.log(v1 / np.sum(list(v.values()))) if v1 != 0 else -3.14e+100 for k1, v1 in v.items()} for k, v in self.B.items()}

            print(self.pi,self.A,self.B)

            # 保存模型
            with open(self.model_file, 'wb') as pkl:
                pickle.dump(self.A, pkl)
                pickle.dump(self.B, pkl)
                pickle.dump(self.pi, pkl)

            return self


    def viterbi(self, text, states, pi, A, B):
        """
        维特比算法实现
        :param text: 待分词文本
        :param states:  状态列表
        :param pi: 初始概率向量
        :param A:   状态转移概率矩阵
        :param B:  观测概率矩阵
        :return:   最大概率,预测状态序列
        """
        delta = [{}] # 不同时刻的前向概率，类似于前缀和
        psi = {}    #记录下来前面已经计算好的路径
        # 初始化delta psi
        for state in states:
            delta[0][state] = pi[state] + B[state].get(text[0], 0)# 从字典 B[state] 中获取 text[0] 对应的值，如果 text[0] 不在 B[state] 中，就返回 0 作为默认值。
            # 第0时刻前向概率，这里本来应该是乘法，但是因为我们前面求了log，所以这里改成加法了
            psi[state] = [state]
        # psi {'B':['B'], 'M':['M'], 'S':['S'], 'E':['E']}

        #todo:第二个字

        '''
        这里计算规则就是：拿第二个字来说
            1.先计算第二个字属于BMSE的概率记作P
            2.假设第二个字属于B，然后计算从第一个字转移的第二个字的的概率
            3.1和2计算出来的值相乘就能得到：如果第二个字属于B它应该从第一个字属于什么转移过来
            
        用球和盒子模型再来理解一下
            从哪个盒子里摸和摸出来哪种颜色都分别对应一个概率，用这两个概率相乘
        '''

        for t in range(1, len(text)):
            delta.append({})
            newpsi = {}

            for state in states:
                '''
                以‘高’这个字举例
                    高这个字可能对应四种状态BMSE
                    我们要把这四种状态中的每一种都取出来，然后计算前面一个字的四种状态到当前状态概率的最大值
                    for state0 in states
                        states：表示‘高’这个字对应的四种状态
                        state0：表示‘高’这个字前面的那个字‘以’对应的四种状态
                    (delta[t-1][state0] + A[state0].get(state, 0), state0)
                        delta[t-1]：先定位到上一个时刻
                        [state0]：再分别遍历上一个时刻的BMSE的四个值
                        A[state0].get(state, 0)：这个是转移概率，就是从上一个时刻转移到当前时刻的概率
                            A[state0]：先定位到上一个时刻
                            get(state：然后转移到当前时刻
                            0)：0代表如果找不到就返回0 
                    求max是
                        比如我们正在算求‘高’这个字对应B时的概率，此时分别算上一个时刻对应BMSE转移到这一时刻B的概率，会得四个值
                        我们取最大的，这就是维特比算法的核心思想
                '''
                (prob, state_sequence) = max([(delta[t-1][state0] + A[state0].get(state, 0), state0) for state0 in states])
                delta[t][state] = prob + B[state].get(text[t], -3.14e100)
                newpsi[state]  = psi[state_sequence] + [state]#保存路径，‘高’映射成B时它前面的词
            # newpsi
            psi = newpsi

        #最后一个时刻概率最大值和对应的路径
        (prob, state_sequence) = max([(delta[len(text)-1][state], state) for state in states])

        return prob, psi[state_sequence]

    def cut(self, text):
        #先把模型加载进来
        if not self.load_param:
            self.try_load_model(os.path.exists(self.model_file))

        #todo:调用维特比算法进行预测
        prob, pos_list = self.viterbi(text, self.state_list, self.pi, self.A, self.B)
        begin, next_idx = 0, 0
        # 分词
        for i, char in enumerate(text):
            pos = pos_list[i] # B M S E中的某个值
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield  text[begin: i+1]
                next_idx = i + 1
            elif pos == 'S':
                yield  char
                next_idx = i + 1

        if next_idx < len(text):
            yield text[next_idx:]

#todo:给文件中的文本分词
def load_article(fname):
    with open(fname, encoding='utf-8') as file:
        aritle = []
        for line in file:
            aritle.append(line.strip())

    return aritle


#todo:辅助模型评估的
def to_region(segmentation):
    """
    把分词结果转成区间
    :param segmentation:  分词后的数据
    :return:  区间
    """
    import re
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end

    return region

#todo:评估指标
def prf(gold, pred):
    """
    计算精确率 召回率 f1
    :param gold: 真实值
    :param pred: 预测值
    :return:
    """
    A, B = set(to_region(gold)), set(to_region(pred))
    A_size = len(A)
    B_size = len(B)
    A_cap_B_size = len(A & B) # TP。A & B真实值和预测值之间的交集，能得到预测对了多少个
    p, r = A_cap_B_size/B_size, A_cap_B_size/A_size#p是查准率，r是查全率
    return p, r, 2*p*r/(p+r)#p,r,F1



if __name__ == '__main__':

    hmm = HMM()
    hmm.try_load_model(True)
    # hmm.train('./HMMTrainSet.txt')

    print(list(hmm.cut('商品和服务')))
    print(list(hmm.cut('项目的研究')))
    print(list(hmm.cut('研究生命起源')))
    print(list(hmm.cut('中文博大精深!')))
    print(list(hmm.cut('这是一个非常棒的方案!')))
    print(list(hmm.cut('武汉市长江大桥')))
    print(list(hmm.cut('普京与川普通话')))
    print(list(hmm.cut('四川普通话')))
    print(list(hmm.cut('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')))
    print(list(hmm.cut('改判被告人死刑立即执行')))
    print(list(hmm.cut('检察院检察长')))
    print(list(hmm.cut('中共中央总书记、国家主席')))


    #todo:预测
    article = load_article('./test1_org.txt')
    pred = "  ".join(list(hmm.cut(article[0])))
    # print(pred)
    gold = load_article('./test1_cut.txt')[0]
    # print(gold)
    print("精确率:%.5f, 召回率:%.5f, F1:%.5f" % prf(gold, pred))

